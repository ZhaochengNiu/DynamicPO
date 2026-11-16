# 探索性实验的偏好优化训练入口，覆盖 MPPO、DynamicPO_MPPO、SDPO、DynamicPO_SDPO 四种目标，由 fire 把 train 暴露为命令行接口
# 承接 sft.py 的监督微调产物，配合 trainer/exploratory_study_trainer.py 验证 DynamicPO 在多种多负样本目标上的泛化与负样本规模扩展
# 导入 os，供拼接输出与 final_checkpoint 路径以及读写环境变量使用
import os
# 导入 random，用于固定随机种子并从候选池采样 neg_num 个负样本
import random
# 导入 re，供模型名与自定义备注的 slug 规范化使用
import re

# 导入 fire，把 train 的参数暴露为命令行选项，供 exploratory_study 下的 .sh 启动
import fire
# 导入 torch，供 bfloat16 计算类型与量化训练相关的张量操作
import torch
# 导入 Accelerator，用 process_index 决定当前进程绑定的 GPU
from accelerate import Accelerator
# 导入 load_dataset，按 json 读取各数据集的 train 与 validation 划分
from datasets import load_dataset
# 从 peft 导入加载 SFT adapter 与 kbit 训练前适配所需的工具
from peft import PeftModel, prepare_model_for_kbit_training
# 从 transformers 导入分词器、训练参数、量化配置与 Llama/Qwen 两类因果模型类
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, LlamaForCausalLM, LlamaTokenizer

# 导入探索性实验自定义 DPOTrainer，实现 MPPO、S-DPO 及其 DynamicPO 版本的损失与动态 beta
from trainer.exploratory_study_trainer import DPOTrainer
# 导入 Prompt，把用户历史与候选物品填充进推荐 prompt 模板
from Prompt import Prompt

# 把 WANDB_MODE 设为 disabled，关闭在线上报，保证探索性实验可离线复现
os.environ["WANDB_MODE"] = "disabled"
# 固定随机种子 1958，使负样本采样与模板选择在复现中一致
random.seed(1958)


# 定义主进程判定，避免多卡训练时重复打印
def _is_main_process() -> bool:
    # RANK 为 0（缺省 0）时视为主进程
    return int(os.environ.get("RANK", "0")) == 0


# 定义把模型路径转成路径友好 slug 的函数，用于输出目录后缀
def _model_name_to_slug(model_name: str) -> str:
    # 取路径末段并去尾部斜杠，空则兜底 model
    base_name = os.path.basename(model_name.rstrip("/")) or "model"
    # 把非字母数字字符折叠为连字符并小写化
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base_name).strip("-").lower()
    # slug 为空时回退 model
    return slug or "model"


# 定义给值追加模型 slug 的函数，隔离不同骨干实验输出
def _append_model_suffix(value: str, model_name: str) -> str:
    # 空值无需追加后缀
    if not value:
        # 原样返回
        return value
    # 生成模型 slug
    slug = _model_name_to_slug(model_name)
    # 已包含则跳过，保持幂等
    if slug in value.lower():
        # 原样返回
        return value
    # 拼接下划线与 slug 得到新值
    return f"{value.rstrip('/')}_{slug}"


# 定义把 custom_note 转成路径友好 slug 的函数
def _note_to_slug(note: str) -> str:
    # 折叠非字母数字字符并小写化
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", note).strip("-").lower()
    # 返回备注 slug
    return slug


# 定义给值追加备注 slug 的函数，区分不同实验配置
def _append_note_suffix(value: str, note: str) -> str:
    # 值或备注为空则跳过
    if not value or not note:
        # 原样返回
        return value
    # 生成备注 slug
    slug = _note_to_slug(note)
    # slug 为空或已存在则跳过
    if not slug or slug in value.lower():
        # 原样返回
        return value
    # 拼接下划线与备注 slug 得到新值
    return f"{value.rstrip('/')}_{slug}"


# 定义 fire 入口 train，集中接收探索性实验的输出、模型、数据与各目标超参
def train(
    # 探索性实验的根输出目录，默认落在 outputs/exploratory_study 下
    output_dir="./outputs/exploratory_study",
    # 基础 LLM 路径或名称，决定 Llama/Qwen 加载分支与输出后缀
    model_name="",
    # prompt 模板路径，决定历史与候选物品的组织方式
    prompt_path="./prompt/music.txt",
    # 数据集名，取 lastfm/goodreads/steam，选择对应 sft-cans20 json
    dataset="lastfm",
    # SFT adapter 路径，作为策略模型与可选参考模型的加载起点
    resume_from_checkpoint="",
    # wandb 运行名，区分不同目标与超参组合
    wandb_name="",
    # 自定义备注，折叠成 slug 后追加到输出目录与运行名
    custom_note="",
    # 偏好损失温度系数 beta，默认 1.0，控制概率比差异的敏感度
    beta: float = 1.0,
    # 损失与筛选模式，取 MPPO、DynamicPO_MPPO、SDPO 或 DynamicPO_SDPO，决定走哪条目标分支
    filter_mode: str = "DynamicPO_MPPO",
    # 附加信息备注，仅主进程打印，便于日志辨认实验
    info_note: str = "",
    # 是否带参考模型，SDPO 系用 w_ref、MPPO 系用 wo_ref，由 .sh 覆盖
    loss_type: str = "w_ref",
    # 每条样本的负样本数，默认 15，扩展实验可取 1/3/5/10/15/19 研究坍缩
    neg_num: int = 15,
    # 单卡训练批量，默认 4
    batch_size: int = 4,
    # 梯度累积步数，默认 8，放大有效批量
    gradient_accumulation_steps: int = 8,
    # 训练轮数，默认 3
    num_train_epochs: int = 3,
    # LoRA 学习率 1e-5，保护 SFT 已学到的兴趣刻画能力
    learning_rate: float = 1e-5,
    # prompt 与响应 token 上限，由 collator 截断控制显存
    cutoff_len: int = 512,
    # 保存与评估步数间隔，.sh 常传 0.033 按训练进度比例触发
    eval_step: int = 4,
    # 是否启用验证集评估，默认关闭
    val_data_eval: bool = False,
# 结束参数列表，函数体开始
):
    # 先给输出目录追加 custom_note 后缀
    output_dir = _append_note_suffix(output_dir, custom_note)
    # 同样给 wandb 名追加 custom_note 后缀
    wandb_name = _append_note_suffix(wandb_name, custom_note)
    # 再给输出目录追加模型 slug，隔离不同骨干
    output_dir = _append_model_suffix(output_dir, model_name)
    # 同样给 wandb 名追加模型 slug
    wandb_name = _append_model_suffix(wandb_name, model_name)

    # 仅主进程打印本次配置，避免多卡重复日志
    if _is_main_process():
        # 打印 custom_note，核对实验备注
        print(f"custom_note: {custom_note}")
        # 打印 info_note，标注运行说明
        print(f"info_note: {info_note}")
        # 打印 beta，确认温度系数
        print(f"beta: {beta}")
        # 打印 filter_mode，确认具体目标分支
        print(f"filter_mode: {filter_mode}")
        # 打印 loss_type，确认是否带参考模型
        print(f"loss_type: {loss_type}")
        # 打印 neg_num，确认负样本规模
        print(f"neg_num: {neg_num}")
        # 打印 batch_size，确认单卡批量
        print(f"batch_size: {batch_size}")
        # 打印梯度累积步数，确认有效批量
        print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
        # 打印 val_data_eval，确认是否构建验证集
        print(f"val_data_eval: {val_data_eval}")
        # 打印输出目录，确认 checkpoint 落盘位置
        print(f"output_dir: {output_dir}")
        # 打印 wandb 运行名，确认实验记录标识
        print(f"wandb_name: {wandb_name}")

    # lastfm 分支，论文推荐的复现数据集
    if dataset == "lastfm":
        # 构造 lastfm 的 train 与 validation 路径字典
        data_files = {
            # 指定 lastfm 训练集 json 路径
            "train": "./data/lastfm-sft-cans20/lastfm-train.json",
            # 指定 lastfm 验证集 json 路径
            "validation": "./data/lastfm-sft-cans20/lastfm-val.json",
        # 闭合 lastfm 路径字典
        }
    # goodreads 分支，提供跨数据集对比
    elif dataset == "goodreads":
        # 构造 goodreads 路径字典
        data_files = {
            # 指定 goodreads 训练集 json 路径
            "train": "./data/goodreads-sft-cans20/goodreads-train.json",
            # 指定 goodreads 验证集 json 路径
            "validation": "./data/goodreads-sft-cans20/goodreads-val.json",
        # 闭合 goodreads 路径字典
        }
    # steam 分支，第三个偏好优化数据集
    elif dataset == "steam":
        # 构造 steam 路径字典
        data_files = {
            # 指定 steam 训练集 json 路径
            "train": "./data/steam-sft-cans20/steam-train.json",
            # 指定 steam 验证集 json 路径
            "validation": "./data/steam-sft-cans20/steam-val.json",
        # 闭合 steam 路径字典
        }
    # 未知数据集进入异常分支
    else:
        # 抛出带数据集名的错误，避免静默误配
        raise ValueError(f"Unsupported dataset: {dataset}")

    # 定义把单条数据字典填充为 Prompt 对象的内部函数
    def convert_dict_to_prompt(d: dict):
        # 按模板路径实例化 Prompt 并读取模板文本
        prompt_builder = Prompt(prompt_path)
        # historyList 为以双冒号分隔的字符串时切成列表，否则保持原样
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        # 把用户历史写入 Prompt，供替换 [HistoryHere]
        prompt_builder.historyList = d["historyList"]
        # 把候选物品列表写入 Prompt，供替换 [CansHere]
        prompt_builder.itemList = d["itemList"]
        # 把真实选择写入 Prompt
        prompt_builder.trueSelection = d["trueSelection"]
        # 返回填充后的 Prompt，其字符串化会随机选模板并拼接占位符
        return prompt_builder

    # 定义 batched 映射，把一批原始样本转成 prompt、chosen 与多个 rejected 列
    def process_data(examples):
        # 初始化输出字典，先放 prompt 与正样本列
        processed = {"prompt": [], "chosen": []}
        # 按 neg_num 预建 rejected1 到 rejectedN 空列
        for i in range(1, neg_num + 1):
            # 为第 i 个负样本建列，供 collator 按 rejected 前缀识别
            processed[f"rejected{i}"] = []

        # 记录原始列名，供 map 删除旧列
        columns = list(examples.keys())
        # 按批内样本数逐条处理
        for i in range(len(examples[columns[0]])):
            # 为当前样本构造包含三要素的字典
            data_point = {
                # 取出第 i 条的真实选择作为正样本
                "trueSelection": examples["trueSelection"][i],
                # 取出第 i 条的候选物品列表，其中含正样本
                "itemList": examples["itemList"][i],
                # 取出第 i 条的用户历史交互
                "historyList": examples["historyList"][i],
            # 闭合当前样本字典
            }
            # 一次性完成模板填充与字符串渲染，得到模型输入
            prompt = str(convert_dict_to_prompt(data_point))
            # 确定 chosen 列为真实选择文本
            chosen = data_point["trueSelection"]
            # 从候选列表剔除 chosen，得到负样本池
            negative_items = [item for item in data_point["itemList"] if item != chosen]
            # 无放回采样 neg_num 个负样本，构成多负样本监督
            sampled_negs = random.sample(negative_items, neg_num)

            # 把渲染后的 prompt 追加到输出列
            processed["prompt"].append(prompt)
            # 把正样本追加到 chosen 列
            processed["chosen"].append(chosen)
            # 带 1 起编号遍历负样本，直接对齐 rejected 列序号
            for idx, rejected in enumerate(sampled_negs, start=1):
                # 把当前负样本写入对应编号的 rejected 列
                processed[f"rejected{idx}"].append(rejected)
        # 返回该批的列式字典，由 datasets.map 汇总
        return processed

    # 按 json 加载所选数据集的 train 与 validation split
    data = load_dataset("json", data_files=data_files)
    # 记录训练集原始列名
    columns = data["train"].column_names

    # 对训练集做 batched 映射，转换为多负样本偏好格式
    train_data = data["train"].map(
        # 传入上面定义的样本转换函数
        process_data,
        # 删除原始列，只保留 prompt、chosen、rejected 等新列
        remove_columns=columns,
        # 8 进程并行加速大规模数据转换
        num_proc=8,
        # 按批处理，使 process_data 收到列式批量字典
        batched=True,
    # 结束映射并按种子 42 打乱训练集，保证可复现
    ).shuffle(seed=42)
    # 先把验证集置空，默认不评估
    val_data = None
    # 仅在开启验证时构建验证集
    if val_data_eval:
        # 对验证 split 做同样的多负样本转换
        val_data = data["validation"].map(
            # 复用同一转换函数，保证训练与验证格式一致
            process_data,
            # 删除验证集原始列
            remove_columns=columns,
            # 同样 8 进程并行
            num_proc=8,
            # 按批处理验证集
            batched=True,
        # 结束映射并以固定种子打乱验证顺序
        ).shuffle(seed=42)
        # 验证集过大时截断以控制评估开销
        if val_data.num_rows > 2000:
            # 取前 2000 条作为验证子集
            val_data = val_data.select(range(2000))

    # 取当前分布式进程编号
    device_index = Accelerator().process_index
    # 把整个模型绑定到当前进程对应的 GPU
    device_map = {"": device_index}
    # 构造 4bit 量化配置，用 QLoRA 放下 7B/8B 骨干
    bnb_config = BitsAndBytesConfig(
        # 启用 4bit 量化加载，降低显存占用
        load_in_4bit=True,
        # 采用 NF4 量化，减小 LLM 权重信息损失
        bnb_4bit_quant_type="nf4",
        # 计算时用 bfloat16 反量化
        bnb_4bit_compute_dtype=torch.bfloat16,
    # 结束量化配置构造
    )

    # Llama-3 与 Qwen 用通用 AutoModelForCausalLM 加载
    if "Llama-3" in model_name or "Qwen" in model_name:
        # 按通用因果模型接口加载待训练骨干
        base_model = AutoModelForCausalLM.from_pretrained(
            # 传入模型路径或名称
            model_name,
            # 绑定到当前进程 GPU
            device_map=device_map,
            # 应用 4bit NF4 量化
            quantization_config=bnb_config,
        # 结束待训练骨干加载
        )
    # 其余 Llama 系列用专用类
    else:
        # 以 Llama 专用类加载 SFT 后骨干
        base_model = LlamaForCausalLM.from_pretrained(
            # 传入模型路径或名称
            model_name,
            # 绑定到当前进程 GPU
            device_map=device_map,
            # 同样 4bit 量化加载
            quantization_config=bnb_config,
        # 结束 Llama 骨干加载
        )
    # 关闭 KV cache，配合梯度检查点避免显存冲突
    base_model.config.use_cache = False
    # 为 kbit 模型做训练前适配，转换归一化层精度、启用梯度检查点并冻结原权重
    base_model = prepare_model_for_kbit_training(base_model)
    # 在量化骨干上加载 SFT 产出的 LoRA adapter 作为策略模型
    base_model = PeftModel.from_pretrained(
        # 传入已适配的量化骨干
        base_model,
        # 传入 SFT adapter 路径
        resume_from_checkpoint,
        # 把 adapter 设为可训练，只更新 LoRA 参数
        is_trainable=True,
    # 结束策略模型加载
    )
    # 仅主进程打印可训练参数统计
    if _is_main_process():
        # 输出 LoRA 参数量，确认冻结策略生效
        base_model.print_trainable_parameters()

    # 决定是否需要参考模型，loss_type 为 wo_ref 或 MPPO 系目标时不需要
    ref_enable = loss_type != "wo_ref" and filter_mode not in {
        # MPPO 目标不依赖参考模型，列入排除集合
        "MPPO",
        # DynamicPO_MPPO 同样不需要参考模型
        "DynamicPO_MPPO",
    # 闭合排除集合，只有 SDPO 系且 w_ref 时才构建参考模型
    }
    # 先置空参考模型，MPPO 系分支直接传 None
    reference_model = None
    # 仅 SDPO 与 DynamicPO_SDPO 且 w_ref 时构建参考模型
    if ref_enable:
        # 参考模型按 Llama-3 与 Qwen 通用类加载
        if "Llama-3" in model_name or "Qwen" in model_name:
            # 加载与策略同构的骨干用于计算参考 logps
            model_ref = AutoModelForCausalLM.from_pretrained(
                # 传入同一模型路径，保证词表一致
                model_name,
                # 绑定到当前进程 GPU
                device_map=device_map,
                # 同样以 4bit 量化加载参考模型
                quantization_config=bnb_config,
            # 结束通用类参考骨干加载
            )
        # 其余 Llama 系列用专用类
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
        # 把同一 SFT adapter 加载到参考骨干，冻结作 DPO 的 KL 锚点
        reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
        # 仅主进程打印参考模型参数统计
        if _is_main_process():
            # 确认锚点模型参数量
            reference_model.print_trainable_parameters()

    # Llama-3 用通用分词器并把 pad 设为 0
    if "Llama-3" in model_name:
        # 加载配套分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # pad token 设为 id 0，配合左填充
        tokenizer.pad_token_id = 0
    # Qwen 需处理 pad token 缺失
    elif "Qwen" in model_name:
        # 加载 Qwen 分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 无专用 pad token 时补齐
        if tokenizer.pad_token is None:
            # 用 eos token 兼作 pad token
            tokenizer.pad_token = tokenizer.eos_token
    # 其余 Llama 系列走 LlamaTokenizer
    else:
        # 加载 Llama 专用分词器
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        # pad token 设为 id 0
        tokenizer.pad_token_id = 0
    # 统一左填充，使生成打分时各序列右端对齐
    tokenizer.padding_side = "left"

    # 集中配置 Trainer 训练超参，供探索性实验 DPOTrainer 使用
    training_args = TrainingArguments(
        # 单卡批量取命令行 batch_size
        per_device_train_batch_size=batch_size,
        # 梯度累积步数取命令行值，放大有效批量
        gradient_accumulation_steps=gradient_accumulation_steps,
        # 启用梯度检查点，用时间换显存
        gradient_checkpointing=True,
        # 梯度裁剪阈值 0.3，抑制多负样本梯度爆炸
        max_grad_norm=0.3,
        # 训练轮数取命令行值
        num_train_epochs=num_train_epochs,
        # 学习率取命令行值，默认 1e-5
        learning_rate=learning_rate,
        # 启用 bfloat16 混合精度，与量化计算 dtype 一致
        bf16=True,
        # 按步数保存 checkpoint
        save_strategy="steps",
        # 保存间隔取 eval_step，.sh 传 0.033 按比例频繁落盘
        save_steps=eval_step,
        # 最多保留 100 个 checkpoint，便于挑选最优步
        save_total_limit=100,
        # 开启验证时按步评估，否则关闭
        evaluation_strategy="steps" if val_data_eval else "no",
        # 评估间隔与保存间隔一致
        eval_steps=eval_step,
        # 不自动回载最优 checkpoint，保留末尾权重
        load_best_model_at_end=False,
        # 每步记录日志，追踪 beta 与奖励训练动态
        logging_steps=1,
        # checkpoint 与日志写入带后缀的输出目录
        output_dir=output_dir,
        # 上报目标设为 wandb，WANDB_MODE 已置 disabled 故实际离线
        report_to="wandb",
        # 运行名用拼接后的 wandb_name
        run_name=wandb_name,
        # 用 bitsandbytes 分页 AdamW，降低优化器显存峰值
        optim="paged_adamw_32bit",
        # 余弦退火学习率，后期逐步收敛
        lr_scheduler_type="cosine",
        # 前 5% 步预热学习率，稳定训练初期
        warmup_ratio=0.05,
        # 保留自定义偏好列，交由 DPODataCollator 处理
        remove_unused_columns=False,
        # 指定可重入梯度检查点，兼容 LoRA 梯度回传
        gradient_checkpointing_kwargs={"use_reentrant": True},
        # 关闭 DDP 未使用参数检测，避免额外开销
        ddp_find_unused_parameters=False,
    # 结束 TrainingArguments 构造
    )

    # 实例化探索性实验 DPOTrainer，传入策略模型、参考模型与配置
    dpo_trainer = DPOTrainer(
        # 位置参数即待优化的 LoRA 策略模型
        base_model,
        # 传入参考模型，MPPO 系为 None、SDPO 系为冻结锚点
        ref_model=reference_model,
        # 传入训练超参
        args=training_args,
        # 传入偏好损失温度系数
        beta=beta,
        # 传入 MPPO、SDPO 及其 DynamicPO 模式，决定损失与动态筛选分支
        filter_mode=filter_mode,
        # 传入 w_ref 或 wo_ref，供 Trainer 判断是否计算参考 logps
        loss_type=loss_type,
        # 传入打乱后的多负样本训练集
        train_dataset=train_data,
        # 仅开启验证时传入验证集
        eval_dataset=val_data if val_data_eval else None,
        # 传入分词器供 collator 分词与填充
        tokenizer=tokenizer,
        # prompt 最大长度取 cutoff_len
        max_prompt_length=cutoff_len,
        # prompt 与响应总长上限同为 cutoff_len
        max_length=cutoff_len,
    # 结束 DPOTrainer 构造
    )

    # 启动偏好优化训练，执行对应目标损失与动态 beta 调整
    dpo_trainer.train()
    # 保存训练得到的 LoRA adapter 到输出目录
    dpo_trainer.save_model(output_dir)

    # 追加 final_checkpoint 子目录存放最终权重
    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    # 把最终策略模型保存到 final_checkpoint
    dpo_trainer.model.save_pretrained(final_output_dir)
    # 把分词器一并保存，便于推理端直接加载
    tokenizer.save_pretrained(final_output_dir)


# 作为脚本直接运行时才触发，被 import 时不执行
if __name__ == "__main__":
    # 用 fire 解析 .sh 传入的参数并启动 train
    fire.Fire(train)
