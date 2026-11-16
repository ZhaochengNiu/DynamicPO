# 本文件是 DynamicPO 全流程的第一阶段：监督微调（SFT）入口，用 4bit 量化加 LoRA 在推荐数据上把基座 LLM 训练出基本的下一物品预测能力
# 产出的 SFT checkpoint（final_checkpoint 下的 adapter 与 tokenizer）供后续 DynamicPO.py 偏好优化阶段作为策略模型与参考模型起点
# 通过 fire.Fire(train) 暴露命令行接口，由 scripts/01_sft/sft.sh 调用
# 导入标准库 os，用于拼接 checkpoint 输出路径、设置 wandb 环境变量以及取出模型路径的最后一段
import os
# 导入 PyTorch，提供 bfloat16 计算类型供 4bit 量化配置使用
import torch
# 导入正则模块 re，用于把模型名规范化成可作目录后缀的 slug
import re
# 导入 wandb，配合 TrainingArguments 的 report_to 记录训练曲线（此处 WANDB_MODE 会被置为 disabled）
import wandb

# 从 transformers 加载因果语言模型、分词器、训练参数与 bitsandbytes 量化配置，分别用于建模型、切 token、组装训练超参和 4bit 量化
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
# 导入 load_dataset，以 json 格式读取 SFT 的 train 与 validation 数据文件
from datasets import load_dataset
# 导入 TRL 的 SFTTrainer 与「仅对回答部分计损失」整理器，让模型只学习 completion 而不对齐 prompt
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# 导入 PEFT 组件：LoraConfig 配置低秩适配器、get_peft_model 注入适配器、prepare_model_for_kbit_training 做 kbit 量化训练前置处理
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# 再次从 transformers 引入 AutoModelForCausalLM 与 LlamaTokenizer，后者用于 Llama-2 等非 Llama-3/Qwen 基座加载
from transformers import AutoModelForCausalLM, LlamaTokenizer

# 导入 Accelerator，用其 process_index 确定当前进程把模型放到哪张 GPU
from accelerate import Accelerator

# 导入顶层 Prompt 再导出模块里的推荐模板类，负责把历史与候选列表拼成模型输入
from Prompt import Prompt

# 重复导入 torch（沿用原文件写法），供下方 bfloat16 等张量相关配置引用
import torch
# 导入 bitsandbytes 并简写为 bnb，支撑 4bit 量化权重加载与分页优化器
import bitsandbytes as bnb
# 导入 fire，用于把 train 函数自动暴露成命令行可调用接口
import fire

# 再次导入 os（沿用原文件写法），下方环境变量设置与路径拼接仍依赖它
import os
# 把 WANDB_MODE 设为 disabled 关闭在线上传，避免本地或多卡实验强依赖 wandb 联网
os.environ['WANDB_MODE']='disabled'
# 导入 random，用于设置全局随机种子以及 Prompt 模板的随机选取
import random
# 固定全局随机种子 1958，保证模板抽样与数据打乱可复现，与偏好优化阶段保持一致
random.seed(1958)


# 定义工具函数：把基座模型路径转换成简短、可作目录名后缀的 slug
def _model_name_to_slug(model_name: str) -> str:
    # 去掉路径末尾斜杠后取最后一段作为基础名，空路径时回退为 "model"
    base_name = os.path.basename(model_name.rstrip("/")) or "model"
    # 把非字母数字字符压成连字符、去首尾连字符并转小写，得到安全的文件名片段
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base_name).strip("-").lower()
    # 若清洗后为空则回退 "model"，保证返回始终是非空可用后缀
    return slug or "model"


# 定义工具函数：给输出目录或 wandb 运行名追加模型 slug，避免不同基座实验结果互相覆盖
def _append_model_suffix(value: str, model_name: str) -> str:
    # 待处理值为空时无需追加，直接进入原值返回分支
    if not value:
        # 原样返回空值，不对空路径做拼接
        return value
    # 先把模型路径规范化成 slug，作为区分不同基座的后缀
    slug = _model_name_to_slug(model_name)
    # 若值里已包含该 slug，说明已带过模型标识，避免重复追加
    if slug in value.lower():
        # 已带后缀则原样返回，防止模型名重复
        return value
    # 去掉尾部斜杠后用下划线拼接 slug，生成带模型标识的最终目录名
    return f"{value.rstrip('/')}_{slug}"

# 定义 SFT 主函数，所有超参以默认参数形式给出并由 fire 通过命令行覆盖
def train(
        
    # path
    # 训练输出根目录，保存 checkpoint 与最终 adapter，稍后自动追加模型 slug
    output_dir="",
    # 训练日志目录，传给 TrainingArguments 记录日志
    logging_dir="",
    # 基座 LLM 的路径或 Hugging Face 名称，决定加载哪个模型与分词器分支
    model_name ="",
    # 推荐 Prompt 模板文件路径（如 prompt/music.txt），Prompt 类从中随机取模板
    prompt_path = "",
    # 数据集名称（lastfm 或 goodreads），用于选择对应的 train 与 validation json
    dataset="",
    # 预留的断点续训或 adapter 加载参数，SFT 主流程未直接使用，保留接口与行尾说明
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    # wandb config
    # wandb 项目名，写入 WANDB_PROJECT 环境变量供训练上报使用
    wandb_project: str = "",
    # 本次 wandb 运行名，稍后追加模型 slug 以区分不同基座（行尾说明原样保留）
    wandb_name: str = "",   # the name of the wandb run
    # training hyperparameters
    # 梯度累积步数，默认 1，即每个 batch 直接反传，可由命令行放大等效 batch
    gradient_accumulation_steps: int = 1,
    # 每张 GPU 的训练 batch 大小，SFT 阶段默认 8，与偏好优化阶段取值不同
    batch_size: int = 8,
    # 训练轮数，SFT 用 5 轮让模型先掌握基本的下一物品预测能力
    num_train_epochs: int = 5,
    # 学习率 2e-5，LoRA 微调常用量级，SFT 略高于偏好优化阶段的 1e-5
    learning_rate: float = 2e-5,
    # 最大序列长度 512，作为 SFTTrainer 的 max_seq_length 截断 prompt 加回答
    cutoff_len: int = 512,
    # 评估与保存步长比例 0.05（按训练总步数比例），同时用于 save_steps 与 eval_steps
    eval_step = 0.05,  

# 结束 train 的参数列表，下面进入函数体
):
    # 给输出目录追加模型 slug，保证不同基座 checkpoint 分目录存放
    output_dir = _append_model_suffix(output_dir, model_name)
    # 给 wandb 运行名追加模型 slug，便于在面板里区分实验
    wandb_name = _append_model_suffix(wandb_name, model_name)

    # 把项目名写进环境变量，使 TrainingArguments 的 report_to 能归到对应 wandb 项目
    os.environ['WANDB_PROJECT'] = wandb_project
    # 定义内层函数：把一条原始样本字典装填进 Prompt 对象
    def convert_dict_to_prompt(d:dict):
        # 用模板文件构造 Prompt 实例，内部会读取并缓存全部候选模板
        t = Prompt(prompt_path)
        # 若历史是以双冒号分隔的字符串则切成列表，已是列表则保持，兼容两种数据格式
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        # 把用户历史交互列表赋给模板对象，用于替换 [HistoryHere]
        t.historyList = d["historyList"]
        # 把 20 个候选物品列表赋给模板对象，用于替换 [CansHere]
        t.itemList = d["itemList"]
        # 记录真实下一物品，SFT 中作为 completion 目标（Prompt 的字符串渲染本身不拼它）
        t.trueSelection = d["trueSelection"]
        # 返回装填好的 Prompt 对象，其字符串渲染会随机产出完整输入文本
        return t

    # 定义单样本处理函数：把一条样本转成 SFTTrainer 需要的 prompt 与 completion 字典
    def process_data(data_point):
        # 先把原始字段装填为 Prompt 对象
        t = convert_dict_to_prompt(data_point)
        # 调用字符串渲染随机选一条模板并填入历史与候选，得到以 Answer 提示结尾的输入
        prompt = str(t)
        # 取出真实的下一物品名称作为监督信号 completion
        target = data_point["trueSelection"]
        # 构造返回字典，字段与 TRL 的 completion-only 约定对齐
        dic = {
            # 输入侧文本：含历史、候选与 Answer 提示
            "prompt": prompt,
            # 输出侧文本：模型应生成的正确物品名
            "completion": target
        # 结束返回字典
        }
        # 返回含 prompt 与 completion 的字典，供 map 逐样本套用
        return dic
    
    # 按数据集名选择 SFT 数据：LastFM（音乐）分支
    if dataset=="lastfm":
        # 构造 load_dataset 需要的 split 到文件映射
        data_files = {
            # LastFM 训练集，每行含 historyList、itemList 与 trueSelection
            "train":"./data/lastfm-sft-cans20/lastfm-train.json",
            # LastFM 验证集，传给 SFTTrainer 的 eval_dataset
            "validation": "./data/lastfm-sft-cans20/lastfm-val.json",
        # 结束 lastfm 分支的数据文件字典
        }
    
    # Goodreads（图书）数据分支
    elif dataset=="goodreads":
        # 构造 Goodreads 的 train 与 validation 映射
        data_files = {
            # Goodreads 训练集路径
            "train":"./data/goodreads-sft-cans20/goodreads-train.json",
            # Goodreads 验证集路径
            "validation": "./data/goodreads-sft-cans20/goodreads-val.json",
        # 结束 goodreads 分支的数据文件字典
        }

    # 又一个同名 goodreads 分支（沿用原文件写法），因前序分支已匹配，实际不会进入
    elif dataset == "goodreads":
         # 该分支的映射使用单数目录名 goodread（缩进多一个空格，照原样保留）
         data_files = {
            # 指向单数拼写的训练文件，作为历史遗留路径保留
            "train": "./data/goodread-sft-cans20/goodread-train.json",
            # 指向单数拼写的验证文件，作为历史遗留路径保留
            "validation": "./data/goodread-sft-cans20/goodread-val.json",
        # 结束该冗余分支的数据文件字典
        }
         
    
    # 用 json reader 载入 train 与 validation 两个 split 成 DatasetDict
    data = load_dataset("json", data_files=data_files)

    # 训练集用 seed 42 打乱后逐样本套 process_data，得到 prompt 与 completion
    train_data = data["train"].shuffle(seed=42).map(process_data)
    # 删掉原始列只留 map 新生成的列，避免 SFTTrainer 收到多余字段
    train_data = train_data.remove_columns(data["train"].column_names)

    # 打印第一条训练样本，便于人工核对 prompt 渲染是否正确
    print(train_data[0])

    # 验证集同样打乱并转成 prompt 与 completion，供 eval_dataset 使用
    val_data = data["validation"].shuffle(seed=42).map(process_data)



    # 配置 bitsandbytes 4bit 量化，让 7B 或 8B 基座能在单卡上以 LoRA 方式微调
    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        # 以 4bit 加载权重，大幅降低显存占用（替代上方被注释掉的 load_in_8bit 选项）
        load_in_4bit=True,
        # 使用 nf4 正态浮点 4bit 量化类型，对预训练权重分布更友好
        bnb_4bit_quant_type="nf4",
        # 计算时反量化到 bfloat16，兼顾数值稳定与速度
        bnb_4bit_compute_dtype=torch.bfloat16,
        # 关闭双重量化（对量化常数再量化），这里选择不启用
        bnb_4bit_use_double_quant=False,
    # 结束量化配置
    )

    # 取当前分布式进程编号，作为本进程绑定的 GPU 序号
    device_index = Accelerator().process_index
    # 把整模型映射到该 GPU，多卡时每进程各占一张卡
    device_map = {"": device_index}

    # 以自回归因果语言模型方式加载基座权重
    base_model = AutoModelForCausalLM.from_pretrained(
        # 基座模型路径或名称
        model_name,
        # 按上面的 device_map 放置到指定 GPU
        device_map=device_map,
        # 应用 4bit 量化配置
        quantization_config=bnb_config,
    # 结束基座模型加载
    )
    # 针对 Llama-3 基座打印标识，便于日志确认所用模型
    if "Llama-3" in model_name:
        # 输出 Llama-3 提示
        print("Llama-3!")
    # 针对 Qwen 基座打印标识
    elif "Qwen" in model_name:
        # 输出 Qwen 提示
        print("Qwen!")
    # 训练时关闭 KV cache，避免与梯度检查点冲突并省显存
    base_model.config.use_cache = False
    # 为 kbit 量化训练做前置处理（冻结基座、cast 层、开梯度检查点等），之后才能挂 LoRA
    base_model = prepare_model_for_kbit_training(base_model)

    # Llama-3 用 AutoTokenizer 加载，pad token 单独处理
    if "Llama-3" in model_name:
        # 加载与 Llama-3 匹配的分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 把 pad token id 设为 0，补全时使用该 id
        tokenizer.pad_token_id = 0
        # 打印所用分词器分支
        print("Llama-3!")
    # Qwen 同样用 AutoTokenizer，但 pad token 可能缺失
    elif "Qwen" in model_name:
        # 加载与 Qwen 匹配的分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 检查 Qwen 是否没有定义 pad token
        if tokenizer.pad_token is None:
            # 缺失时复用 eos token 作为 pad token，避免批量补全报错
            tokenizer.pad_token = tokenizer.eos_token
        # 打印所用分词器分支
        print("Qwen!")
    # 其余基座（如 Llama-2）走 LlamaTokenizer 分支
    else:
        # 用 LlamaTokenizer 加载 LLaMA 系分词器
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        # 把 pad token id 设为 0
        tokenizer.pad_token_id = 0
        
    
    # 统一在左侧补全，便于批量生成时右对齐取新 token（行尾说明：规避 fp16 溢出问题）
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training

    # Change the LORA hyperparameters accordingly to fit your use case
    # 配置 LoRA 低秩适配器，只训练少量增量权重以节省显存
    peft_config = LoraConfig(
        # 关闭推理模式，使适配器权重可训练
        inference_mode=False,
        # 低秩秩 r 取 8，控制新增可训练参数量
        r=8,
        # 缩放系数 alpha 取 32，与 r 共同决定增量权重的放大倍数
        lora_alpha=32,
        # 把 LoRA 注入注意力与 MLP 的全部线性层，覆盖 Transformer 主要可适配位置
        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        # LoRA 层 dropout 取 0.1，缓解小数据 SFT 过拟合
        lora_dropout=0.1,
        # 声明为因果语言建模任务，使 PEFT 采用对应输出头与包装
        task_type="CAUSAL_LM",
    # 结束 LoRA 配置
    )

    # 把 LoRA 适配器注入量化基座，得到可训练的 PeftModel
    base_model = get_peft_model(base_model, peft_config)

    # 定义 SFTTrainer 的格式化函数，把批样本拼成完整训练文本
    def formatting_prompts_func(example):
        # 初始化列表，收集每条拼接后的文本
        output_texts = []
        # 遍历 batch 内每条样本
        for i in range(len(example['prompt'])):
            # 把 prompt、completion 与 eos 直接串成一条文本，让模型学到回答后结束
            text = f"{example['prompt'][i]}{example['completion'][i]}{tokenizer.eos_token}"
            # 收集该条完整文本
            output_texts.append(text)
        # 返回整批文本列表供分词
        return output_texts
    
    # 以模板末尾的 Answer 提示作为回答起始标记
    response_template = "Answer:"
    # 把 Answer 标记编码后去掉首 token 作为分隔模板，整理器据此把 prompt 部分 label 屏蔽、只对 completion 计损失
    collator = DataCollatorForCompletionOnlyLM(tokenizer.encode(response_template, add_special_tokens = False)[1:], tokenizer=tokenizer)


    # 组装 HF Trainer 的全部训练超参
    training_args = TrainingArguments(
        # 每卡训练 batch，来自命令行参数 batch_size
        per_device_train_batch_size=batch_size,
        # 梯度累积步数，放大等效 batch
        gradient_accumulation_steps=gradient_accumulation_steps,
        # 开启梯度检查点，用计算换显存以适配大模型
        gradient_checkpointing =True,
        # 梯度裁剪阈值 0.3，抑制 4bit 加 LoRA 训练中的梯度爆炸
        max_grad_norm= 0.3,
        # 训练轮数，来自命令行
        num_train_epochs=num_train_epochs, 
        # 学习率，来自命令行
        learning_rate=learning_rate,
        # 使用 bfloat16 混合精度，与 4bit 计算 dtype 一致
        bf16=True,
        # 按步数保存 checkpoint
        save_strategy="steps",
        # 保存步长取 eval_step（比例 0.05），周期性落盘
        save_steps=eval_step,
        # 最多保留 100 个 checkpoint，便于回溯不同阶段权重
        save_total_limit=100,
        # 不在训练末尾自动加载最优权重，避免与后续手动保存冲突
        load_best_model_at_end=False,
        # SFT 阶段不做验证集评估，仅产出供偏好优化用的权重
        evaluation_strategy="no",
        # 评估步长仍设为 eval_step（因不评估而实际不触发）
        eval_steps=eval_step,
        # 每步记录一次日志，便于观察 SFT loss 收敛
        logging_steps=1,
        # checkpoint 与最终模型写到带模型后缀的目录
        output_dir=output_dir,
        # 使用分页 AdamW 优化器，显存峰值更低，配合 4bit 量化
        optim="paged_adamw_32bit",
        # 余弦学习率衰减，训练后期更稳定
        lr_scheduler_type="cosine",
        # 前 5% 步数做 warmup，防止初始大梯度破坏量化权重
        warmup_ratio=0.05,
        # 训练指标上报到 wandb（前面已设 WANDB_MODE 为 disabled）
        report_to="wandb",
        # 使用带模型后缀的运行名
        run_name=wandb_name,
        # 日志落盘目录
        logging_dir=logging_dir,
        # 指定使用 reentrant 版梯度检查点，兼容 PEFT 与量化模型的梯度回传
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        # 只保存模型权重不存优化器状态，显著减小 checkpoint 体积
        save_only_model=True,
        # 关闭 DDP 未使用参数检测（行尾说明：前向没有未使用参数），加快分布式训练
        ddp_find_unused_parameters=False, # should set to False becuase there are no unused parameters in the forward process
    # 结束训练参数配置
    )

    # 用 TRL 的 SFTTrainer 组装监督微调流程
    trainer = SFTTrainer(
        # 挂好 LoRA 的可训练策略模型
        base_model,
        # 已转成 prompt 与 completion 的训练集
        train_dataset=train_data,
        # 已转格式的验证集（本配置不做评估）
        eval_dataset=val_data,
        # 使用 completion-only 整理器，只对齐回答部分计损失
        data_collator=collator,
        # 分词器，负责编码与补全
        tokenizer=tokenizer,
        # 最大序列长度，截断过长的 prompt 加回答
        max_seq_length=cutoff_len,
        # 用自定义函数把样本拼成完整训练文本
        formatting_func=formatting_prompts_func,
        # 传入上面组装好的训练超参
        args=training_args
    # 结束 SFTTrainer 构造
    )

    # 启动监督微调训练循环
    trainer.train() 
    # 把训练得到的 LoRA adapter 保存到输出目录
    trainer.save_model(output_dir)

    # 在输出目录下再建 final_checkpoint 子目录，存放最终权重
    output_dir = os.path.join(output_dir, "final_checkpoint")
    # 保存最终 PeftModel 权重，供 DynamicPO.py 作为 SFT checkpoint 加载
    trainer.model.save_pretrained(output_dir)
    # 一并保存分词器，保证偏好优化与推理阶段用同一套分词
    tokenizer.save_pretrained(output_dir)

# 作为脚本直接运行时才触发下面的入口
if __name__ == "__main__":
    # 用 fire 把 train 暴露为命令行接口，scripts/01_sft/sft.sh 通过命令行参数调用
    fire.Fire(train)
