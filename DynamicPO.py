# 主实验 DMPO 与 DynamicPO_DMPO 的偏好优化训练入口，由 fire 把 train 暴露为命令行接口供 scripts 下的 .sh 调用
# 承接 sft.py 的监督微调产物，本脚本构建多负样本偏好数据并交给 trainer/dynamicpo_trainer.py 的 DPOTrainer 完成优化
# 导入 os，供拼接输出与 checkpoint 路径、设置 WANDB_MODE 环境变量以及读取分布式 RANK 使用
import os
# 导入 torch，供 bfloat16 计算类型、张量运算以及 4bit 量化训练相关的底层操作
import torch
# 导入 re，供把模型路径与自定义备注规范化为可作目录名的 slug
import re
# 导入 random，用于固定随机种子以及从候选池采样 neg_num 个负样本
import random
# 导入 numpy 并以 np 为别名，供数据与训练过程中的数值计算使用
import numpy as np
# 从 peft 导入 LoRA 配置、模型包装与状态字典工具，用于加载并微调 SFT 后的 adapter
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
# 从 transformers 导入分词器、训练参数、因果语言模型与量化配置，覆盖 Llama 与 Qwen 两类骨干
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer
# 导入 load_dataset，用于按 json 读取 lastfm/goodreads/steam 的训练与验证划分
from datasets import load_dataset
# 导入主实验自定义 DPOTrainer，它实现 DMPO 与 DynamicPO_DMPO 的损失及动态负样本筛选
from trainer.dynamicpo_trainer import DPOTrainer
# 再次从 peft 导入 kbit 训练准备函数，为 4bit 量化模型做训练前的精度与梯度适配
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# 再次导入通用因果模型与分词器类，供按骨干名称分支加载模型时使用
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入 Prompt（trainer.utils.Prompt 的再导出），把用户历史与候选物品填充进 prompt 模板
from Prompt import Prompt
# 导入 json，供偏好数据以 json 文件形式读写时使用的底层解析能力
import json
# 重复导入 torch，使量化配置与张量运算共用同一模块级 torch 命名空间
import torch
# 导入 bitsandbytes 并以 bnb 为别名，为 4bit NF4 量化与分页优化器提供底层算子
import bitsandbytes as bnb
# 导入 Accelerator，用其 process_index 决定当前进程把模型放到哪张 GPU
from accelerate import Accelerator
# 导入 fire，把 train 的所有参数自动暴露为命令行选项，供 .sh 脚本直接传参
import fire

# 导入 matplotlib 绘图接口，保留原实现引入的绘图能力
import matplotlib.pyplot as plt
# 再次导入 os，供 os.path.join 拼接最终 checkpoint 目录等路径操作使用
import os 
# 把 WANDB_MODE 设为 disabled，避免训练时向云端上报，保证实验离线可复现
os.environ["WANDB_MODE"] = 'disabled'
# 固定 Python 随机种子为 1958，使负样本采样与 prompt 模板选择在多进程复现中一致
random.seed(1958)


# 定义主进程判定函数，用分布式 RANK 是否为 0 避免多卡训练时重复打印与保存
def _is_main_process() -> bool:
    # 读取环境变量 RANK（缺省 0）并与 0 比较，仅主进程返回 True
    return int(os.environ.get("RANK", "0")) == 0


# 定义把模型路径转成路径友好短标识的函数，用于给输出目录与 wandb 名追加骨干后缀
def _model_name_to_slug(model_name: str) -> str:
    # 取模型路径的最后一段并去掉尾部斜杠，空值时兜底为 model
    base_name = os.path.basename(model_name.rstrip("/")) or "model"
    # 把非字母数字字符折叠成连字符、去首尾连字符并转小写，得到可作目录名的 slug
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base_name).strip("-").lower()
    # slug 为空时回退为 model，保证后缀始终非空
    return slug or "model"


# 定义给路径或名称追加模型 slug 后缀的函数，避免不同骨干实验互相覆盖输出
def _append_model_suffix(value: str, model_name: str) -> str:
    # 待处理值为空则无需追加后缀
    if not value:
        # 直接返回原值，保持空值语义
        return value
    # 由模型名生成规范化 slug
    slug = _model_name_to_slug(model_name)
    # 若值里已包含该 slug 则不再重复追加，保持幂等
    if slug in value.lower():
        # 返回未改动的值
        return value
    # 去掉尾部斜杠后拼接下划线与 slug，得到带骨干标识的新值
    return f"{value.rstrip('/')}_{slug}"


# 定义把自定义备注转成路径友好短标识的函数
def _note_to_slug(note: str) -> str:
    # 把备注中的非字母数字字符折叠为连字符并小写化
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", note).strip("-").lower()
    # 返回规范化后的备注 slug
    return slug


# 定义给路径或名称追加 custom_note 后缀的函数，用于区分不同实验配置
def _append_note_suffix(value: str, note: str) -> str:
    # 值为空或备注为空都无需追加后缀
    if not value or not note:
        # 直接返回原值
        return value
    # 把备注规范化为 slug
    slug = _note_to_slug(note)
    # slug 为空或已包含在值中则跳过，避免无效或重复后缀
    if not slug or slug in value.lower():
        # 返回未改动的值
        return value
    # 拼接下划线与备注 slug，得到带实验标识的新值
    return f"{value.rstrip('/')}_{slug}"


# 定义 fire 命令行入口 train，集中接收输出路径、模型、数据、DPO 超参以及损失与筛选模式
def train(
    #train
    # 模型与 checkpoint 的根输出目录，默认当前目录，运行时由 .sh 脚本覆盖
    output_dir="./",
    # 基础 LLM 的路径或名称，决定加载 Llama/Qwen 分支并参与输出目录后缀
    model_name ="",
    # prompt 模板文件路径，决定如何把用户历史与候选物品组织成模型输入
    prompt_path = "./prompt/music.txt",
    # 数据集名，取 lastfm/goodreads/steam，用于选择对应的 sft-cans20 json 划分
    dataset="lastfm",
    # SFT checkpoint 或最终 adapter 路径，作为待优化 LoRA 适配器与参考模型的加载起点
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter
    # wandb 运行名，用于区分不同超参组合的实验记录
    wandb_name: str = "",   # the name of the wandb run
    # 自定义备注，会折叠成 slug 追加到输出目录与 wandb 名以便归档
    custom_note:str="",
    # training hyperparameters
    # DPO 损失温度系数 beta，控制偏好对概率比差异的敏感度，主实验默认 1.0
    beta: float = 1.0,
    
    # 损失与筛选模式，主实验取 DMPO 或 DynamicPO_DMPO，决定 trainer 走哪条损失分支
    filter_mode:str='',
    # 附加信息备注，仅在主进程打印，便于实验日志辨认
    info_note:str='',
    # 是否带参考模型，w_ref 使用参考模型、wo_ref 不使用，主实验固定 w_ref
    loss_type:str="w_ref",
    # 每条样本采样的负样本数，主实验默认 15，扩展实验可到 19 研究偏好优化坍缩
    neg_num: int = 15,
    # 每张 GPU 的训练批大小，配合梯度累积放大有效批量
    batch_size: int = 4,
    # 梯度累积步数，4 卡下有效批量随之放大，稳定多负样本训练
    gradient_accumulation_steps: int = 8,
    # 训练轮数，默认 3 轮以充分优化又不至于过拟合偏好数据
    num_train_epochs: int = 3,
    # LoRA 微调学习率 1e-5，偏好优化阶段用较小学习率保护 SFT 已学到的能力
    learning_rate: float = 1e-5,
    # prompt 与响应 token 的最大长度，超出由 collator 截断以控制显存
    cutoff_len: int = 512,
    # 保存与评估的步数间隔，.sh 常传 0.033 表示按训练进度比例触发
    eval_step = 4,
    # 是否启用验证集评估，默认关闭以节省开销，开启时才构建 val_data
    val_data_eval: bool = False,
# 结束 train 的参数列表，函数体随后开始
):
    # 先把 custom_note 后缀追加到输出目录，使不同备注的实验落在独立目录
    output_dir = _append_note_suffix(output_dir, custom_note)
    # 同样把 custom_note 后缀追加到 wandb 运行名
    wandb_name = _append_note_suffix(wandb_name, custom_note)
    # 再给输出目录追加模型 slug，区分不同骨干 LLM 的结果
    output_dir = _append_model_suffix(output_dir, model_name)
    # 同样给 wandb 名追加模型 slug，保证记录不覆盖
    wandb_name = _append_model_suffix(wandb_name, model_name)

    # 仅主进程打印配置，规避多卡 torchrun 下的重复日志
    if _is_main_process():
        # 打印 custom_note，便于日志核对实验备注
        print(f"custom_note:{custom_note}")
        # 打印 info_note，标注本次运行的附加说明
        print(f"info_note:{info_note}")
        # 打印 beta，确认 DPO 温度系数取值
        print(f"beta: {beta}")
        # 打印 filter_mode，确认走 DMPO 还是 DynamicPO_DMPO 分支
        print(f"filter_mode: {filter_mode}")
        # 打印 loss_type，确认是否使用参考模型
        print(f"loss_type:{loss_type}")
        # 打印 neg_num，确认每条样本的负样本数量
        print(f"neg_num: {neg_num}")
        # 打印 batch_size，确认单卡批量
        print(f"batch_size: {batch_size}")
        # 打印梯度累积步数，确认有效批量大小
        print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
        # 打印 val_data_eval，确认是否构建验证集
        print(f"val_data_eval:{val_data_eval}")
        # 打印拼接后的输出目录，确认 checkpoint 落盘位置
        print(f"output_dir:{output_dir}")
        # 打印拼接后的 wandb 运行名，确认实验记录标识
        print(f"wandb_name:{wandb_name}")

    # 按数据集名分支，lastfm 是论文推荐的复现首选
    if dataset=="lastfm":
        # 构造 lastfm 的 train 与 validation 文件路径字典
        data_files = {
            # 指定 lastfm SFT 候选集 cans20 的训练集 json 路径
            "train":"./data/lastfm-sft-cans20/lastfm-train.json",
            # 指定 lastfm 验证集 json 路径，供开启验证时采样
            "validation": "./data/lastfm-sft-cans20/lastfm-val.json",
        # 闭合 lastfm 的 data_files 字典
        }
    # goodreads 分支，提供跨数据集泛化对比
    elif dataset=="goodreads":
        # 构造 goodreads 的 train 与 validation 路径字典
        data_files = {
            # 指定 goodreads 训练集 json 路径
            "train":"./data/goodreads-sft-cans20/goodreads-train.json",
            # 指定 goodreads 验证集 json 路径
            "validation": "./data/goodreads-sft-cans20/goodreads-val.json",
        # 闭合 goodreads 的 data_files 字典
        }
    # steam 分支，第三个多负样本偏好优化数据集
    elif dataset=="steam":
        # 构造 steam 的 train 与 validation 路径字典
        data_files = {
            # 指定 steam 训练集 json 路径
            "train":"./data/steam-sft-cans20/steam-train.json",
            # 指定 steam 验证集 json 路径
            "validation": "./data/steam-sft-cans20/steam-val.json",
        # 闭合 steam 的 data_files 字典
        }
    # 未匹配到支持的数据集时进入异常分支
    else:
        # 抛出带数据集名的错误，避免拼写错误导致的静默错误配置
        raise ValueError(f"Unsupported dataset: {dataset}")

    # 定义把单条数据字典填充为 Prompt 对象的内部函数
    def convert_dict_to_prompt(d:dict):
        # 按模板路径实例化 Prompt，读取并缓存模板文本
        t = Prompt(prompt_path)
        # historyList 为以双冒号分隔的字符串时切成列表，已是列表则保持原样
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        # 把用户历史交互写入 Prompt，用于替换模板中的 [HistoryHere]
        t.historyList = d["historyList"]
        # 把候选物品列表写入 Prompt，用于替换 [CansHere]
        t.itemList = d["itemList"]
        # 把真实选择（正样本物品）写入 Prompt
        t.trueSelection = d["trueSelection"]
        # 返回填充好的 Prompt 对象，其字符串化会随机选模板并拼接占位符
        return t

    # 定义 batched 映射函数，把一批原始样本转成 prompt、chosen 与多个 rejected 列
    def process_data(examples):
        # 初始化输出字典，先放 prompt 与正样本两列
        dic = {"prompt":[], "chosen":[]}
        # 按 neg_num 预建 rejected1 到 rejectedN 的空列表列
        for i in range(1, neg_num+1):
            # 为第 i 个负样本创建对应列，供 DPOTrainer 按 rejected 前缀识别
            dic[f"rejected{i}"] = []
        # 记录原始列名，供外层 map 的 remove_columns 使用
        columns = list(examples.keys())
        # 按批内样本数逐条遍历，batched 下 examples 为列式字典
        for i in range(len(examples[columns[0]])):
            # 为当前样本构造单条字典
            data_point = {}
            # 取出第 i 条的真实选择作为正样本
            data_point["trueSelection"] = examples["trueSelection"][i]
            # 取出第 i 条的候选物品列表，其中含正样本
            data_point["itemList"] = examples["itemList"][i]
            # 取出第 i 条的用户历史交互
            data_point["historyList"] = examples["historyList"][i]  
            # 把单条数据填充进 Prompt 模板对象
            t = convert_dict_to_prompt(data_point)
            # 渲染得到最终输入文本，随机模板并替换历史与候选占位符
            prompt = str(t)
            # 确定 chosen 列为真实选择物品文本
            chosen = data_point["trueSelection"]
            # 从候选列表中剔除正样本，得到可采样的负样本池
            negative_items = [item for item in data_point["itemList"] if item != data_point["trueSelection"]]
            # 无放回随机采样 neg_num 个负样本，构成多负样本偏好监督
            sample_negs = random.sample(negative_items, neg_num)
            # 把渲染后的 prompt 追加到输出列
            dic["prompt"].append(prompt)
            # 把正样本追加到 chosen 列
            dic["chosen"].append(chosen)
            # 初始化负样本序号计数器，用于生成 rejected1 起的列名
            cnt = 0  
            # 逐个遍历采样得到的负样本
            for rejected in sample_negs:
                # 序号自增，与 rejected 列编号对齐
                cnt += 1
                # 把当前负样本写入对应编号的 rejected 列
                dic[f"rejected{cnt}"].append(rejected)
        # 返回该批转换后的列式字典，由 datasets.map 汇总成新列
        return dic


    # 按 json 加载所选数据集的 train 与 validation 两个 split
    data = load_dataset("json", data_files=data_files)

    # 记录训练集原始列名，供 map 删除旧列
    columns = data["train"].column_names
    # 用 process_data 对训练集做 batched 映射并删除原始列，8 进程并行后按种子 42 打乱，末尾以反斜杠把调用延续到下一行
    # 紧随其后的续行给出 num_proc=8 与 batched=True，并以 shuffle(seed=42) 固定打乱顺序；行尾另附一段停用的调试截断写法，两者由反斜杠连成同一条语句
    train_data = data["train"].map(process_data, remove_columns=columns, \
                                    num_proc=8, batched=True).shuffle(seed=42)#.select(range(256))
    # 仅主进程打印样本预览
    if _is_main_process():
        # 预览前 4 条转换后的训练样本
        for i in range(0, 4):
            # 打印第 i 条样本，检查 prompt、chosen 与 rejected 是否符合预期
            print(train_data[i])

    # 先把验证集置空，默认不评估
    val_data = None
    # 仅在开启验证集评估时构建 val_data
    if val_data_eval:
        # random 2000 samples for validation
        # 验证集同样用 process_data 做 8 进程 batched 映射、删旧列并按种子 42 打乱，末尾以反斜杠把调用延续到下一行
        # 紧随其后的续行给出 num_proc=8 与 batched=True，并以 shuffle(seed=42) 固定验证集顺序保证评估可复现，两者由反斜杠连成同一条语句
        val_data = data["validation"].map(process_data, remove_columns=columns, \
                                            num_proc=8, batched=True).shuffle(seed=42)
        # 验证集过大时截断，控制评估开销
        if val_data.num_rows > 2000:
            # 取前 2000 条作为验证子集，实现随机 2000 条验证的近似采样
            val_data = val_data.select(range(2000))
    
   # print(val_data)


    # 取当前分布式进程编号，决定模型放置的 GPU
    device_index = Accelerator().process_index
    # 构造 device_map，把整个模型绑定到当前进程对应的 GPU，配合 torchrun 多卡
    device_map = {"": device_index}
        
    # 构造 4bit 量化配置，用 QLoRA 方式在单卡放下 7B/8B 骨干
    bnb_config = BitsAndBytesConfig(
        # 启用 4bit 量化加载，显著降低显存占用
        load_in_4bit=True,
        # 采用 NF4 正态浮点量化，对 LLM 权重分布信息损失更小
        bnb_4bit_quant_type="nf4",
        # 计算时用 bfloat16 反量化，兼顾数值范围与显存
        bnb_4bit_compute_dtype=torch.bfloat16,
    # 结束量化配置的构造
    )


    # Llama-3 与 Qwen 使用通用 AutoModelForCausalLM 分支加载
    if "Llama-3" in model_name or "Qwen" in model_name:
        # 按通用因果语言模型接口加载待训练骨干
        base_model = AutoModelForCausalLM.from_pretrained(
            # 传入模型路径或名称
            model_name,
            # 按进程编号把模型放到对应 GPU
            device_map=device_map,
            # 应用 4bit NF4 量化配置
            quantization_config=bnb_config,
        # 结束待训练模型的加载
        )
    # 其余 Llama 系列走专用 LlamaForCausalLM 分支
    else:
        # 用 Llama 专用类加载 SFT 后的骨干
        base_model = LlamaForCausalLM.from_pretrained(
            # 传入模型路径或名称
            model_name,
            # 绑定到当前进程 GPU
            device_map=device_map,
            # 同样应用 4bit 量化
            quantization_config=bnb_config,
        # 结束 Llama 骨干加载
        )
    # 关闭 KV cache，梯度检查点训练时避免缓存与显存冲突
    base_model.config.use_cache = False
    # 为 kbit 量化模型做训练前适配，转换归一化层精度、启用梯度检查点并冻结原权重
    base_model = prepare_model_for_kbit_training(base_model)
    # 在量化骨干上加载 SFT 产出的 LoRA adapter 作为待优化策略模型
    base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint,#)#, 
                                        # 把加载的 adapter 设为可训练，使 DPO 只更新 LoRA 参数
                                        is_trainable=True)
    # 仅主进程打印可训练参数统计
    if _is_main_process():
        # 输出可训练 LoRA 参数量，确认冻结策略生效
        base_model.print_trainable_parameters()

    # 由 loss_type 决定是否构建参考模型，w_ref 需要、wo_ref 不需要
    ref_enable=loss_type not in ["wo_ref"]
    # 需要参考模型时进入构建分支
    if ref_enable:
        # 参考模型按 Llama-3 与 Qwen 通用类分支加载
        if "Llama-3" in model_name or "Qwen" in model_name:
            # 加载与策略模型同构的骨干用于计算参考 logps
            model_ref = AutoModelForCausalLM.from_pretrained(
                # 传入同一模型路径，保证参考与策略词表一致
                model_name,
                # 绑定到当前进程 GPU
                device_map=device_map,
                # 同样以 4bit 量化加载参考模型节省显存
                quantization_config=bnb_config,
            # 结束通用类参考骨干加载
            )
        # 其余 Llama 系列用专用类分支
        else:
            # 以 Llama 专用类加载参考骨干
            model_ref = LlamaForCausalLM.from_pretrained(
                # 传入同一模型路径
                model_name,
                # 绑定到当前进程 GPU
                device_map=device_map,
                # 应用 4bit 量化配置
                quantization_config=bnb_config,
            # 结束 Llama 参考骨干加载
            )
        # 把同一 SFT adapter 加载到参考骨干，冻结作为 DPO 的 KL 锚点
        reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
        # 仅主进程打印参考模型参数统计
        if _is_main_process():
            # 确认参考模型参数量，核对锚点模型已就绪
            reference_model.print_trainable_parameters()



    # Llama-3 使用通用分词器并把 pad 设为 0
    if "Llama-3" in model_name:
        # 加载与骨干配套的分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 把 pad token 设为 id 0，配合左填充批量推理
        tokenizer.pad_token_id = 0
    # Qwen 分支需处理 pad token 缺失
    elif "Qwen" in model_name:
        # 加载 Qwen 分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Qwen 若无专用 pad token 时补齐
        if tokenizer.pad_token is None:
            # 用 eos token 兼作 pad token，避免填充报错
            tokenizer.pad_token = tokenizer.eos_token
    # 其余 Llama 系列走 LlamaTokenizer 分支
    else:
        # 加载 Llama 专用分词器
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        # 同样把 pad token 设为 id 0
        tokenizer.pad_token_id = 0
    # 统一左填充，使生成式打分时各序列右端对齐，便于取末端 token 的 logits
    tokenizer.padding_side = "left"  

    # 集中配置 HuggingFace Trainer 的训练超参，供 DPOTrainer 使用
    training_args = TrainingArguments(
        # 单卡批量取命令行 batch_size，默认 4
        per_device_train_batch_size=batch_size,
        # 梯度累积步数取命令行值，默认 8，放大有效批量
        gradient_accumulation_steps=gradient_accumulation_steps,
        # 启用梯度检查点，用时间换显存以容纳长 prompt 与多负样本
        gradient_checkpointing =True,
        # 梯度裁剪阈值 0.3，抑制多负样本损失带来的梯度爆炸
        max_grad_norm= 0.3,
        # 训练轮数取命令行值，默认 3
        num_train_epochs=num_train_epochs, 
        # 学习率取命令行值，默认 1e-5
        learning_rate=learning_rate,
        # 启用 bfloat16 混合精度，与 4bit 计算 dtype 一致并加速训练
        bf16=True,
        # 按步数保存 checkpoint，便于按训练进度取中间 adapter
        save_strategy="steps",
        # 保存间隔取 eval_step，.sh 传 0.033 表示按比例频繁落盘
        save_steps=eval_step,
        # 最多保留 100 个 checkpoint，便于后续推理挑选最优步
        save_total_limit=100,
        # 开启验证时按步评估，否则关闭评估以省开销
        evaluation_strategy="steps" if val_data_eval else "no",
        # 评估间隔与保存间隔一致，便于对齐保存点与指标
        eval_steps=eval_step,
        # 不自动回载最优 checkpoint，保留训练末尾权重供最终 adapter 使用
        load_best_model_at_end=False,
        # 每步记录一次日志，便于追踪 beta 与奖励等训练动态
        logging_steps=1,
        # checkpoint 与日志写入带备注与模型后缀的输出目录
        output_dir=output_dir,
        # 上报目标设为 wandb，WANDB_MODE 已置 disabled 故实际离线
        report_to = "wandb",
        # 运行名用拼接后的 wandb_name 区分实验
        run_name = wandb_name,
        # 使用 bitsandbytes 分页 AdamW，显存峰值时对优化器状态做分页
        optim="paged_adamw_32bit",
        # 余弦退火学习率，训练后期逐步收敛
        lr_scheduler_type="cosine",
        # 前 5% 步做学习率预热，稳定量化模型训练初期
        warmup_ratio=0.05,
        # 保留 prompt、chosen、rejected 等自定义列，交由 DPODataCollator 处理
        remove_unused_columns=False,
        # 指定可重入梯度检查点，兼容 PEFT 与 LoRA 的梯度回传
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        # 关闭 DDP 未使用参数检测，避免多负样本分支下的额外开销与告警
        ddp_find_unused_parameters=False,
    # 结束 TrainingArguments 构造
    )

    # 实例化主实验自定义 DPOTrainer，传入策略模型、参考模型与全部配置
    dpo_trainer = DPOTrainer(
        # 第一个位置参数即待优化的 LoRA 策略模型
        base_model,
        # 按 ref_enable 决定是否传入冻结的参考模型，wo_ref 时为 None
        ref_model=reference_model if ref_enable else None,
        # 传入上面构造的训练超参
        args=training_args,
        # 传入 DPO 温度系数 beta
        beta=beta,
        # 传入 DMPO 或 DynamicPO_DMPO 模式，决定损失与动态负样本筛选逻辑
        filter_mode=filter_mode,
        # 传入打乱后的多负样本训练集
        train_dataset=train_data,
        # 仅在开启验证时传入验证集
        eval_dataset=val_data if val_data_eval else None,
        # 传入分词器供 DPODataCollator 做分词与填充
        tokenizer=tokenizer,
        # prompt 最大长度取 cutoff_len，超长由 collator 截断
        max_prompt_length=cutoff_len,
        # prompt 与响应总长上限同为 cutoff_len，控制显存
        max_length=cutoff_len
    # 结束 DPOTrainer 构造
    )



    # 启动偏好优化训练循环，执行 DMPO 或 DynamicPO 损失与动态 beta 调整
    dpo_trainer.train()
    # 保存训练得到的 LoRA adapter 到输出目录
    dpo_trainer.save_model(output_dir)

    # 在输出目录下追加 final_checkpoint 子目录，存放最终可部署权重
    output_dir = os.path.join(output_dir, "final_checkpoint")
    # 把最终策略模型 LoRA 权重保存到 final_checkpoint
    dpo_trainer.model.save_pretrained(output_dir)
    # 把分词器一并保存，保证推理端可直接加载完整配置
    tokenizer.save_pretrained(output_dir)

# 作为脚本直接运行时才触发入口，被 import 时不执行
if __name__ == "__main__":
    # 用 fire 把 train 暴露为命令行，解析 .sh 传入的参数并启动训练
    fire.Fire(train)
