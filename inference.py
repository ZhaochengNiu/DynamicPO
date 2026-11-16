# 本文件是 DynamicPO 全流程的第三阶段：推理入口，加载基座模型与训练好的 LoRA adapter，在测试集上生成下一物品推荐文本
# 把渲染好的 prompt 交给 evaluate_batch.evaluate 逐批生成回答并统计 HitRatio@1 与有效回答比例
# 由 scripts/03_inference/inference.sh 经 fire.Fire(inference) 调用，是产出论文指标的评估入口
# 导入 load_dataset，以 json 格式读取测试集 split
from datasets import load_dataset
# 从 transformers 加载因果语言模型、Llama 分词器、4bit 量化配置与自动分词器，覆盖不同基座的加载需求
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoTokenizer
# 导入顶层 Prompt 再导出类，用于把测试样本的历史与候选列表渲染成输入 prompt
from Prompt import Prompt
# 导入 torch，用于取 bfloat16 计算类型并承载生成张量
import torch
# 导入 DataLoader（沿用原文件写法），评估阶段按 batch 组织输入时可用
from torch.utils.data import DataLoader
# 导入 transformers（沿用原文件写法），提供生成相关的配置类型
import transformers
# 导入 evaluate_batch 的 evaluate 函数，推理末尾直接调用它完成批量生成与指标统计
from evaluate_batch import evaluate
# 导入 PeftModel 以加载 LoRA adapter，prepare_model_for_kbit_training 在本文件保留导入
from peft import PeftModel, prepare_model_for_kbit_training
# 导入 Accelerator，用 process_index 决定模型放到哪张 GPU
from accelerate import Accelerator
# 导入 fire，把 inference 函数暴露成命令行接口
import fire



# 定义推理主函数，所有参数给默认值并由 fire 通过命令行覆盖
def inference( dataset="",
               # 评估时每批样本数，传给 evaluate 控制生成吞吐
               batch_size: int = 0,
               # 训练好的 LoRA adapter 路径（SFT 或偏好优化产出的 checkpoint），非空时挂载到基座
               resume_from_checkpoint: str = "",
               # 基座 LLM 路径或名称，同时用于选择分词器分支
               base_model = "",
               # 外部 Prompt 模板路径，为空时回退到默认 prompt.txt
               external_prompt_path = "",
               # 结束 inference 参数列表
               ):
    # 把 base_model 自赋值（沿用原文件写法），保持变量名供后续加载使用
    base_model = base_model
    # 打印所用基座模型，便于日志确认推理加载的对象
    print(base_model)
    # 取 torch.bfloat16 作为量化计算 dtype，与训练阶段保持一致
    compute_dtype = getattr(torch, "bfloat16")
    # 配置 4bit 量化加载，使大模型能在单卡上完成生成
    bnb_config = BitsAndBytesConfig(
        # 以 4bit 加载权重，降低推理显存占用
        load_in_4bit=True,
        # 使用 nf4 量化类型，与训练时保持一致以免精度偏差
        bnb_4bit_quant_type="nf4",
        # 计算 dtype 用上面取到的 bfloat16
        bnb_4bit_compute_dtype=compute_dtype,
        # 关闭双重量化，与训练配置对齐
        bnb_4bit_use_double_quant=False,
        # load_in_8bit=True,
    # 结束量化配置（块内 load_in_8bit 为被注释的备选，原样保留）
    )
    # 取当前进程编号作为绑定 GPU 序号
    device_index = Accelerator().process_index
    # 把整模型映射到该 GPU
    device_map = {"": device_index}
    # 以自回归因果语言模型方式加载基座权重
    model = AutoModelForCausalLM.from_pretrained(
        # 基座模型路径或名称
        base_model,
        # 按 device_map 放到指定 GPU
        device_map=device_map,
        # 应用 4bit 量化配置
        quantization_config=bnb_config,
    # 结束基座模型加载
    )

    # 若指定了 adapter 路径，则把训练好的 LoRA 权重挂到基座上
    if resume_from_checkpoint != "":
        # 用 PeftModel.from_pretrained 加载 SFT 或偏好优化产出的 adapter，得到最终评测模型
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
    # 切换到 eval 模式，关闭 dropout 等训练专属行为
    model.eval()

    # Llama-3 走 AutoTokenizer 分支
    if "Llama-3" in base_model:
        # 加载与 Llama-3 匹配的分词器
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        # 把 pad token id 设为 0，批量左补全时使用
        tokenizer.pad_token_id = 0
    # Qwen 走 AutoTokenizer 分支
    elif "Qwen" in base_model:
        # 加载与 Qwen 匹配的分词器
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        # 检查 Qwen 是否缺少 pad token
        if tokenizer.pad_token is None:
            # 缺失时复用 eos token 作为 pad token
            tokenizer.pad_token = tokenizer.eos_token
    # 其余基座（如 Llama-2）走 LlamaTokenizer 分支
    else:
        # 用 LlamaTokenizer 加载 LLaMA 系分词器
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        # 把 pad token id 设为 0
        tokenizer.pad_token_id = 0

    # 统一左侧补全，保证批量生成时各样本右对齐、新生成 token 位置一致
    tokenizer.padding_side = "left"
    

    
    # 定义内层函数：把一条测试样本装填进 Prompt 对象
    def convert_dict_to_prompt(d:dict):
        # 用模板文件构造 Prompt 实例（原文件此处变量名后有两个空格，照原样保留）
        t  = Prompt(prompt_path)
        # 若历史是双冒号分隔的字符串则切成列表，兼容两种数据格式
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        # 把用户历史列表赋给模板对象，用于替换 [HistoryHere]
        t.historyList = d["historyList"]
        # 把候选物品列表赋给模板对象，用于替换 [CansHere]
        t.itemList = d["itemList"]
        # 记录真实下一物品，供 evaluate 判定命中（渲染本身不拼它）
        t.trueSelection = d["trueSelection"]
        # 返回装填好的 Prompt 对象
        return t
    
    # 定义单样本处理函数：把测试样本转成带 prompt 字段的字典
    def generate_and_tokenize_prompt(data_point):
        # 先把原始字段装填为 Prompt 对象
        t = convert_dict_to_prompt(data_point)
        # 调用字符串渲染随机选模板并填入历史与候选
        prompt = str(t)
        # 直接复用原始样本字典，保留 trueSelection 与 itemList 供评估
        dic = data_point
        # 把渲染文本去掉末尾空格后存入 prompt 字段，避免模板尾部空格干扰生成
        dic["prompt"] = prompt[:-1]
        # 返回含 prompt 的样本字典，供 map 与 evaluate 使用
        return dic
    

    # 确定模板路径：未指定外部路径时用默认 prompt.txt，否则用外部路径
    prompt_path = "prompt.txt" if external_prompt_path == "" else external_prompt_path
    # LastFM（音乐）测试集分支
    if dataset == "lastfm":
        # 构造只含 test split 的数据文件映射
        data_files = {
            # LastFM 测试集 json 路径
            "test": "./data/lastfm-sft-cans20/lastfm-test.json"
        # 结束 lastfm 分支的数据文件字典
        }
    # Goodreads（图书）测试集分支
    elif dataset == "goodreads":
        # 构造只含 test split 的数据文件映射
        data_files = {
            # Goodreads 测试集 json 路径
            "test": "./data/goodreads-sft-cans20/goodreads-test.json"
        # 结束 goodreads 分支的数据文件字典
        }
    # Steam（游戏）测试集分支
    elif dataset == "steam":
        # 构造只含 test split 的数据文件映射
        data_files = {
            # Steam 测试集 json 路径
            "test": "./data/steam-sft-cans20/steam-test.json"
        # 结束 steam 分支的数据文件字典
        }
    # 数据集名不在支持列表时进入异常分支
    else:
        # 抛出带数据集名的 ValueError，避免静默使用空 data_files
        raise ValueError(f"Unsupported dataset: {dataset}")


    # 用 json reader 载入 test split
    data = load_dataset("json", data_files=data_files)
    # 清理 datasets 缓存文件，避免旧缓存影响本次推理读取
    data.cleanup_cache_files()
    # 打印 DatasetDict 概览，便于确认测试集规模
    print(data)

    # 对测试集逐样本生成 prompt 字段，得到可直接评估的数据
    test_data = data["test"].map(generate_and_tokenize_prompt)
   
    # 调用 evaluate 批量生成回答并统计准确率（HitRatio@1）与有效回答比例
    accuracy, valid_ratio = evaluate(model, tokenizer, test_data, batch_size=batch_size)
    # 打印最终准确率与有效率，即论文报告的核心指标
    print(accuracy, valid_ratio)
    


# 作为脚本直接运行时才触发下面的入口
if __name__ == "__main__":
    # 用 fire 把 inference 暴露为命令行接口，由 scripts/03_inference/inference.sh 调用
    fire.Fire(inference)
