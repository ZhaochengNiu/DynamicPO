# 本文件是 DynamicPO 全流程的评估模块，被 inference.py 调用，对测试集逐批做贪心生成并统计 HitRatio@1 与有效回答比例
# 核心逻辑：把模型生成文本与候选物品列表比对，只有恰好命中一个候选才算有效回答，其中命中真实下一物品才计入得分
# 针对 Llama-3 与 Qwen 采用 token 级切片、其余基座采用字符串前缀裁剪，以适配不同分词器对 prompt 的回显差异
# 导入 torch（沿用原文件写法），承载张量与设备相关操作
import torch

# 导入 transformers（沿用原文件写法），提供生成配置等类型
import transformers
# 导入 List 类型标注，供函数签名注解使用
from typing import List
# 导入 load_dataset（沿用原文件写法），本模块主要通过传入的 val_data 取数
from datasets import load_dataset
# 导入 json（沿用原文件写法），保留原文件导入
import json
# 从 transformers 加载 LlamaForCausalLM、LlamaTokenizer 与 GenerationConfig，后者用于配置贪心生成
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig
# 导入 PeftModel（沿用原文件写法），适配挂载了 LoRA 的评测模型
from peft import PeftModel
# 导入 torch.nn（沿用原文件写法），保留原文件导入
import torch.nn as nn
# 导入 DataLoader（沿用原文件写法），保留原文件导入
from torch.utils.data import DataLoader
# 导入 random（沿用原文件写法），保留原文件导入
import random
# 导入 fire 的 Fire（沿用原文件写法），保留原文件导入
from fire import Fire
# 导入 tqdm，为逐批评估循环显示进度条
from tqdm import tqdm

# 设定默认 device_map 为 auto（沿用原文件写法），模块级变量保留
device_map = "auto"
# 定义评估主函数 evaluate，接收模型、分词器、验证数据与批大小
def evaluate(
    # 待评估的模型（可含 LoRA adapter），负责生成回答
    model,
    # 与模型配套的分词器，负责编码与解码
    tokenizer,
    # 已带 prompt、trueSelection、itemList 字段的测试数据集
    val_data,
    # 每批样本数，默认 32，由 inference.py 传入覆盖
    batch_size: int = 32
# 结束 evaluate 参数列表
):
    # 优先取分词器的 name_or_path，缺失时回退到模型 config 的路径，用于识别基座类型
    model_name = getattr(tokenizer, "name_or_path", "") or getattr(model.config, "_name_or_path", "")
    # 把模型名转小写，便于大小写无关地判断是否 Llama-3 或 Qwen
    normalized_model_name = model_name.lower()
    # Llama-3 与 Qwen 的 tokenizer 会把特殊 token 一并解码，故需用 token 级切片而非字符串裁剪
    use_token_level_slice = "llama-3" in normalized_model_name or "qwen" in normalized_model_name
    
    # 定义内层生成函数，把一批 prompt 转成模型回答文本列表
    def output_generate(
        # 本批待生成的 prompt 文本列表
        prompts,
        # 采样温度参数（本评估固定贪心，实际未启用）
        temperature = 0,
    # 结束 output_generate 参数列表
    ):
        # print([len(prompt) for prompt in prompts])
        # 用分词器把整批 prompt 编码为 pt 张量，开启截断与左补全、上限 1024，并移到模型所在设备
        inputs = tokenizer(prompts,return_tensors="pt",truncation=True,padding=True,max_length=1024).to(model.device)
        # 构造生成配置，控制解码方式
        generation_config = GenerationConfig(
            # temperature = temperature,
            # 关闭采样使用贪心解码，保证评估确定性（块内 temperature 为被注释的备选，原样保留）
            do_sample = False,
        # 结束生成配置
        )
        # 调用模型 generate 批量自回归生成回答
        generation_output = model.generate(
            # 展开上面编码得到的 input_ids 与 attention_mask
            **inputs,
            # 显式传入 pad token id，避免批量生成时的补全告警
            pad_token_id = tokenizer.pad_token_id,
            # 应用上面构造的贪心生成配置
            generation_config = generation_config,
            # 以字典形式返回生成结果，便于取 sequences 与 scores
            return_dict_in_generate = True,
            # 同时输出每步 scores（本评估主要用 sequences）
            output_scores = True,
            # 最多生成 20 个新 token，足够覆盖一个候选物品名
            max_new_tokens = 20
        # 结束 generate 调用
        )
        # 根据基座类型选择回答截取方式
        if use_token_level_slice:
            # 记录输入 prompt 的 token 数，作为切分生成序列的边界
            prompt_token_count = inputs["input_ids"].shape[1]
            # 按 token 切片，只保留 prompt 之后新生成的 token，避免把回显的 prompt 当回答
            generated_tokens = generation_output.sequences[:, prompt_token_count:]
            # 把新生成 token 批量解码成文本，跳过特殊 token
            output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # 其余基座走字符串裁剪分支
        else:
            # 直接解码完整序列（含 prompt 回显），随后按前缀长度裁剪
            output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        # 对每条回答去首尾空白，便于后续精确比对候选名
        output = [_.strip() for _ in output]
        # 返回本批清洗后的回答文本列表
        return output
    
    # 初始化真实目标列表，收集每条样本的 trueSelection
    targets = []
    # 初始化输入 prompt 列表，收集每条渲染好的 prompt
    inputs = []
    # 初始化候选列表，收集每条样本的 20 个候选物品
    cans = []
    # 遍历测试数据集每条样本
    for elm in val_data:
        # 取出该样本渲染好的输入 prompt
        prompt = elm["prompt"]
        # 取出该样本真实的下一物品名称，作为命中判定基准
        target = elm["trueSelection"]
        # 把真实目标追加到 targets
        targets.append(target)
        # 把 prompt 追加到 inputs
        inputs.append(prompt)
        # 把候选物品列表追加到 cans，供有效性判定
        cans.append(elm["itemList"])

    # 用向上取整公式算总批数，保证最后不足一批的样本也被覆盖
    batch_num = (len(inputs)-1)// batch_size + 1
    # 初始化命中得分计数（回答命中真实物品的次数）
    score = 0
    # 初始化有效回答计数（回答中恰好含一个候选物品的次数）
    valid = 0
    # 按批循环评估，用 tqdm 显示进度
    for i in tqdm(range(batch_num), desc="Testing..."):
        # 本批起始下标为批号乘批大小
        start = i*batch_size
        # 本批结束下标取输入总数与 start 加 batch_size 的较小值，防止越界
        end = min(len(inputs), start+batch_size)
        # 切出本批 prompt 文本
        batch_inputs = inputs[start:end]
        # 调用 output_generate 得到本批模型回答
        outputs = output_generate(batch_inputs)
        # 切出本批真实目标
        batch_targets = targets[start:end]
        # 切出本批候选物品列表
        batch_cans = cans[start:end]
        # 把本批的 prompt、回答、目标、候选逐条配对遍历
        for input_text, output, target, candidates in zip(batch_inputs, outputs, batch_targets, batch_cans):
            # Llama-3 与 Qwen 采用前面已 token 级切片的结果
            if use_token_level_slice:
                # 回答文本已经是纯生成部分，直接作为 selection
                selection = output
            # 其余基座需按字符串前缀裁剪
            else:
                # 用输入 prompt 的字符长度裁掉回显前缀，得到纯生成回答
                selection = output[len(input_text):]
            # 统计候选物品中有多少个出现在回答里，用于判定是否为有效单选回答
            num_cans = sum([1 for can in candidates if can in selection])  
            # 打印输入 prompt，便于人工核对评估样本
            print(input_text)
            # 打印候选物品列表，便于核对选项
            print(candidates)
            # 打印模型截取后的回答，便于观察生成质量
            print(selection)
            # 打印真实目标，便于直观对照是否命中
            print([target])
            # 只有回答里恰好命中一个候选物品时才视为有效回答
            if num_cans == 1:
                # 有效回答计数加一
                valid += 1
                # 在有效前提下进一步判断真实目标是否出现在回答中
                if target in selection:
                    # 命中则得分加一，累计 HitRatio@1 的分子
                    score += 1
                    # 打印当前累计得分，便于实时观察指标增长
                    print(f"score increased to {score}")
            # 打印空行，分隔每条样本的调试输出
            print("\n")

            
    # 返回命中率（得分除以样本总数）与有效率（有效回答数除以样本总数）
    return score/len(inputs), valid/len(inputs)
