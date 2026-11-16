# 本文件是 DynamicPO 偏好优化阶段的公用工具：Prompt 模板类、DPO 数据整理器与张量补齐函数
# Prompt 把用户历史与候选物品渲染成训练或推理输入；DPODataCollatorWithPadding 把 chosen 与多个 rejected 分词、屏蔽 prompt 的 label 并补齐成批
# 被 sft.py、DynamicPO.py、inference.py 间接依赖，是连接数据到 Trainer 的关键中间层
# 导入标准库 os，用于判断 Prompt 模板文件是否存在
import os
# 导入 random，用于设置全局随机种子以及 Prompt 模板的随机选取
import random
# 导入 warnings（沿用原文件写法），保留原文件导入
import warnings
# 导入 dataclass 装饰器，用于以声明式字段定义 DPODataCollatorWithPadding
from dataclasses import dataclass
# 导入 Any、Dict、List、Optional、Union 等类型标注，供字段与函数签名注解
from typing import Any, Dict, List, Optional, Union

# 导入 numpy（沿用原文件写法），保留原文件导入
import numpy as np
# 导入 torch，提供 LongTensor、张量拼接与张量类型
import torch
# 导入 pad_sequence，用于把一个 batch 内不等长序列补齐到相同长度
from torch.nn.utils.rnn import pad_sequence
# 导入 IterableDataset（沿用原文件写法），保留原文件导入
from torch.utils.data import IterableDataset
# 从 transformers 导入语言建模整理器基类、分词器基类与 TrainerCallback 类型（沿用原文件写法）
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, TrainerCallback

# 固定全局随机种子 1958，保证 Prompt 模板抽样与训练入口一致、可复现
random.seed(1958)

# 定义 Prompt 模板类，把推荐样本的历史与候选列表渲染为模型输入文本
class Prompt:
    # 定义构造方法，接收模板文件路径
    def __init__(self, prompt_path) -> None:
        # 断言模板文件必须存在，否则明确提示需要指定 Prompt 模板
        assert os.path.isfile(prompt_path), "Please specify a prompt template"
        # 以只读文本方式打开模板文件
        with open(prompt_path, 'r') as f:
            # 按行读取全部模板并去掉行尾换行，每行是一条可选 Prompt 模板
            raw_prompts = f.read().splitlines()
        # 对每条模板去首尾空白后缓存到 self.templates，供随机抽样
        self.templates = [p.strip() for p in raw_prompts]
            
        # 初始化用户历史交互列表为空，等待外部赋值
        self.historyList = []
        # 初始化候选物品列表为空，等待外部赋值
        self.itemList = []
        # 初始化真实下一物品为空字符串，等待外部赋值
        self.trueSelection = ""

    # 定义字符串渲染方法，使对 Prompt 实例取字符串时得到一条填好占位符的完整输入
    def __str__(self) -> str:
        # 从缓存的模板里随机选一条，实现训练时 Prompt 多样性以增强鲁棒性
        prompt = self.templates[random.randint(0, len(self.templates)-1)]
        # 把历史交互列表用逗号空格连成字符串，用于替换占位符
        history = ", ".join(self.historyList)
        # 把候选物品列表用逗号空格连成字符串，用于替换占位符
        cans = ", ".join(self.itemList)
        # 把模板里的历史占位符替换为真实历史
        prompt = prompt.replace("[HistoryHere]", history)
        # 把模板里的候选占位符替换为候选物品列表
        prompt = prompt.replace("[CansHere]", cans)
        # 在末尾补一个空格，使以 Answer 结尾的提示后留出生成位置
        prompt += " "

            
        # 返回渲染完成的输入文本，SFT 与推理阶段据此构造 prompt
        return prompt
    
    
# 用 dataclass 装饰器把下面的类字段声明自动转成构造方法
@dataclass
# 定义 DPO 数据整理器：把 chosen 与多个 rejected 分词、屏蔽 prompt 的 label 并补齐成批，供 DPOTrainer 使用
class DPODataCollatorWithPadding:
    # 类文档说明：本整理器把批内各序列补齐到最大长度，其参数逐条解释如下
    # tokenizer：用于编码数据的分词器，决定 input_ids、attention_mask 与 pad/eos id
    # padding：传给分词器的补齐策略，默认 True 表示按批内最长补齐
    # max_length：单条序列（prompt 加回答）允许的最大长度，超出会触发截断
    # max_prompt_length：prompt 部分允许的最大长度，截断时先压缩 prompt 以尽量保留回答
    # label_pad_token_id：用于屏蔽 label 的填充值，默认 -100，交叉熵会忽略该值
    # padding_value：attention_mask 等张量的补齐值，默认 0
    # truncation_mode：prompt 加回答过长时的截断方式，默认 keep_end 保留 prompt 尾部
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt + chosen/rejected responses.
    """
    # 声明字段 tokenizer：分词器实例，负责编码与提供各类特殊 token id
    tokenizer: PreTrainedTokenizerBase
    # 声明字段 padding：补齐策略，默认 True 表示按批内最长序列补齐
    padding: Union[bool, str] = True
    # 声明字段 max_length：整条序列最大长度，默认 None 时由调用方给定
    max_length: Optional[int] = None
    # 声明字段 max_prompt_length：prompt 最大长度，截断时用于限制 prompt 侧
    max_prompt_length: Optional[int] = None
    # 声明字段 label_pad_token_id：默认 -100，交叉熵忽略该值从而不对 prompt 与 padding 计损失
    label_pad_token_id: int = -100
    # 声明字段 padding_value：默认 0，用于 attention_mask 等非 label 张量的补齐
    padding_value: int = 0
    # 声明字段 truncation_mode：默认 keep_end，过长时保留 prompt 靠近回答的尾部
    truncation_mode: str = "keep_end"

    # 定义单元素分词方法：把一条样本的 prompt、chosen 与多个 rejected 转成带掩码的 token 字典
    def tokenize_batch_element(
        # 实例自身作为首个参数
        self,
        # 输入的 prompt 文本（含历史与候选的渲染结果）
        prompt: str,
        # 偏好正例 chosen 文本，即真实的下一物品
        chosen: str,
        # 偏好负例字典，键为 rejected1、rejected2 等，值为对应负样本文本，支撑多负样本 DPO
        rejected: Dict[str, str],
    # 方法返回一个字典，内含各序列的 input_ids、attention_mask 与 labels
    ) -> Dict:
        # 方法文档说明：对单条样本分词并在此阶段只做截断，暂不转 PyTorch 张量
        # 先截断 prompt：若 prompt 加 chosen 或加 rejected 仍超长，再截断 chosen 与 rejected 回答
        # 同时为 chosen 与每个 rejected 创建 labels，长度等于 prompt 与回答长度之和
        # labels 中对应 prompt 的位置填 label_pad_token_id，使损失只作用于回答部分
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        # 对正例 chosen 分词，add_special_tokens 为 False 避免自动插入 bos 或 eos 破坏拼接
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        # 对 prompt 分词，同样不自动加特殊 token，便于与回答直接拼接
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        # 初始化负例分词结果字典，键与 rejected 一致
        rejected_tokens = {}
        # 遍历每个负例键（对应多个负样本）
        for key in rejected:
            # 对当前负样本文本分词，不自动加特殊 token
            rejected_tokens[key] = self.tokenizer(rejected[key], add_special_tokens=False)
            
        # 断言 prompt 不含 eos token，避免提前终止信号混入输入
        assert self.tokenizer.eos_token_id not in prompt_tokens["input_ids"], f"Prompt contains EOS token: {prompt}"
        # 开启多行断言：检查正例回答是否含 eos token
        assert (
            # 断言正例 chosen 的 token 里不含 eos token
            self.tokenizer.eos_token_id not in chosen_tokens["input_ids"]
        # 断言失败时给出含正例文本的错误信息
        ), f"Chosen response contains EOS token: {chosen}"
        # 开启多行断言：检查所有负例回答是否含 eos token
        assert (
            # 用 all 逐个检查每个负例的 token 里都不含 eos token
            all([self.tokenizer.eos_token_id not in rejected_tokens[key]["input_ids"] for key in rejected_tokens])
        # 断言失败时给出含负例字典的错误信息
        ), f"Rejected response contains EOS token: {rejected}"

        # 给正例 chosen 的 input_ids 末尾追加 eos token，标记回答结束
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        # 同步给正例的 attention_mask 追加 1，保证 eos 位置参与注意力
        chosen_tokens["attention_mask"].append(1)
        # 遍历每个负例分词结果
        for key in rejected_tokens:
            # 给当前负例的 input_ids 末尾追加 eos token
            rejected_tokens[key]["input_ids"].append(self.tokenizer.eos_token_id)
            # 同步给当前负例的 attention_mask 追加 1
            rejected_tokens[key]["attention_mask"].append(1)
        # 计算所有负例中最长的 input_ids 长度，用于判断整体是否超长
        max_rejected_len = max([len(rejected_tokens[key]["input_ids"]) for key in rejected_tokens])
        # 取正例长度与最长负例长度的较大值，作为回答侧需要容纳的长度
        longer_response_length = max(len(chosen_tokens["input_ids"]), max_rejected_len)

        # if combined sequence is too long, truncate the prompt
        # 若 prompt 长度加最长回答超过 max_length，进入 prompt 截断分支
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            # keep_start 模式：保留 prompt 开头部分
            if self.truncation_mode == "keep_start":
                # 按 max_prompt_length 截取每个字段的前缀，实现保留开头的截断
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            # keep_end 模式：保留 prompt 结尾部分（更靠近回答，通常更相关）
            elif self.truncation_mode == "keep_end":
                # 按 max_prompt_length 截取每个字段的后缀，实现保留尾部的截断
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            # 截断模式不在支持范围时进入异常分支
            else:
                # 抛出含未知模式名的 ValueError，避免静默错误处理
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        # 若截断 prompt 后仍超长，则进入回答侧截断分支
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            # 按剩余额度（max_length 减 max_prompt_length）截断正例 chosen 各字段
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            # 按同样额度截断所有负例各字段，保证正负例长度一致可比
            rejected_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()}

        # Create labels
        # 把 prompt 与正例回答的 token 逐字段拼接，得到完整 chosen 序列
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        # 初始化负例完整序列字典
        rejected_sequence_tokens = {}
        # rejected_tokens: Dict[str, Dict]
        # 遍历每个负例分词结果
        for key in rejected_tokens:
            # 把 prompt 与当前负例回答逐字段拼接，得到该负例的完整序列
            rejected_sequence_tokens[key] = {k: prompt_tokens[k] + rejected_tokens[key][k] for k in rejected_tokens[key]}
        # 先复制正例完整序列的 input_ids 作为 labels，使回答部分默认参与损失计算
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        # 把 labels 中属于 prompt 的前缀替换成 label_pad_token_id（-100）：prompt 只是条件输入不应作为预测目标，交叉熵忽略 -100，从而只在回答上计损失
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            # len 取 prompt 的 input_ids 长度，决定要屏蔽的 token 个数（括号内隐式续行，可安全插入注释）
            prompt_tokens["input_ids"]
        # 结束 len 调用与赋值，正例的 label 屏蔽完成
        )
        # 遍历每个负例完整序列
        for key in rejected_sequence_tokens:
            # 先复制该负例完整序列的 input_ids 作为 labels
            rejected_sequence_tokens[key]["labels"] = rejected_sequence_tokens[key]["input_ids"][:]
            # 把该负例 labels 中属于 prompt 的一段替换为 label_pad_token_id
            rejected_sequence_tokens[key]["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                # len 的参数同样是 prompt token 数，保证与正例屏蔽长度一致（括号内隐式续行，可安全插入注释）
                prompt_tokens["input_ids"]
            # 结束 len 调用与赋值，所有负例的 prompt 部分 label 屏蔽完成
            )

        # 初始化返回字典 batch，汇总文本字段与分词字段
        batch = {}

        # 保存原始 prompt 文本，供 Trainer 记录或参考模型使用
        batch["prompt"] = prompt
        # 保存 prompt 加正例回答的拼接文本
        batch["chosen"] = prompt + chosen
        # 遍历每个负例键
        for key in rejected:
            # 保存 prompt 加当前负例回答的拼接文本，键沿用 rejected1 等
            batch[key] = prompt + rejected[key]
        # 单独保存正例回答文本（不含 prompt），便于只编码回答
        batch["chosen_response_only"] = chosen
        # 遍历每个负例键
        for key in rejected:
            # 单独保存当前负例回答文本，键形如 rejected1_response_only
            batch[f"{key}_response_only"] = rejected[key]

        # 遍历 chosen 序列与 prompt 序列两类分词结果，准备写入 batch
        for k, toks in {
            # chosen 键对应正例完整序列（含被屏蔽 prompt 的 labels）
            "chosen": chosen_sequence_tokens,
            # "rejected": rejected_sequence_tokens,
            # prompt 键对应单独的 prompt 分词结果，供参考模型计算 prompt 侧
            "prompt": prompt_tokens,
        # 结束字典并调用 items 逐项遍历（rejected 项在此被注释排除，单独在下方处理）
        }.items():
            # 遍历该类序列内的每个字段（input_ids、attention_mask、labels 等）
            for type_key, tokens in toks.items():
                # 跳过 token_type_ids 字段，DPO 计算不需要该字段
                if type_key == "token_type_ids":
                    # 遇到 token_type_ids 直接跳过本轮
                    continue
                # 把该字段按序列名加字段名写入 batch，如 chosen_input_ids、prompt_attention_mask
                batch[f"{k}_{type_key}"] = tokens
        # rejected_sequence_tokens: Dict[str, Dict]
        # 单独遍历每个负例完整序列，键为 rejected1 等
        for k, toks in rejected_sequence_tokens.items():
            # 遍历该负例序列内的每个字段
            for type_key, tokens in toks.items():
                # 同样跳过 token_type_ids 字段
                if type_key == "token_type_ids":
                    # 遇到 token_type_ids 直接跳过本轮
                    continue
                # 把该负例字段按负例键加字段名写入 batch，如 rejected1_input_ids、rejected1_labels
                batch[f"{k}_{type_key}"] = tokens
        
        # 返回组装好的单样本字典，供 collate 补齐成批
        return batch

    # 定义 collate 方法，把多条不等长样本补齐到同一长度
    def collate(self, batch):
        # first, pad everything to the same length
        # 初始化补齐后的批字典
        padded_batch = {}
        # 以第一条样本的键为基准，逐字段处理整个 batch
        for k in batch[0].keys():
            # 只对 input_ids、attention_mask、labels 三类张量字段做补齐
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # adapted from https://stackoverflow.com/questions/73256206
                # 若字段属于 prompt 侧，需要特殊处理成左补齐
                if "prompt" in k:
                    # 把每条 prompt 序列先倒序，使 pad_sequence 补在右侧后再翻回，等效于左侧补齐
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                # 其余字段不做倒序，正常右侧补齐
                else:
                    # 把各样本该字段转成长整型张量列表，供 pad_sequence 使用
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                # input_ids 字段用分词器的 pad_token_id 补齐
                if k.endswith("_input_ids"):
                    # 取分词器 pad_token_id 作为补齐值
                    padding_value = self.tokenizer.pad_token_id
                # labels 字段用 label_pad_token_id（-100）补齐，保证 padding 位置不计损失
                elif k.endswith("_labels"):
                    # 取 label_pad_token_id 作为补齐值
                    padding_value = self.label_pad_token_id
                # attention_mask 字段用 padding_value（默认 0）补齐，标记 padding 位置不参与注意力
                elif k.endswith("_attention_mask"):
                    # 取 padding_value 作为补齐值
                    padding_value = self.padding_value
                # 出现预期外的字段名时进入异常分支
                else:
                    # 抛出含字段名的 ValueError，避免静默忽略未知字段
                    raise ValueError(f"Unexpected key in batch '{k}'")

                # 用 pad_sequence 按 batch_first 补齐到批内最长，使用上面选定的补齐值
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                # 若字段属于 prompt 侧，需要把补齐方向翻回左侧
                if "prompt" in k:
                    # 沿第 1 维翻转，把之前倒序产生的右侧 padding 变回左侧 padding，保证生成时右对齐
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            # 非张量字段（文本字段）不需要补齐
            else:
                # 直接把各样本文本收集成列表存入 batch
                padded_batch[k] = [ex[k] for ex in batch]

        # 返回补齐后的完整批字典
        return padded_batch

    # 定义 __call__，使整理器实例可像函数一样被 DataLoader 调用
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 初始化列表，收集逐条分词后的样本
        tokenized_batch = []

        # 遍历 DataLoader 传入的每条原始特征
        for feature in features:
            # 取出该样本的 prompt 文本
            prompt = feature["prompt"]
            # 取出该样本的正例 chosen 文本
            chosen = feature["chosen"]
            # 初始化负例字典
            rejected = {}
            # 遍历该样本的所有键
            for key in feature:
                # 筛出以 rejected 开头的键，对应多负样本字段
                if key.startswith("rejected"):
                    # 把该负例文本收进 rejected 字典，键沿用 rejected1 等
                    rejected[key] = feature[key]

            # 调用单元素分词方法，得到含屏蔽 label 的 token 字典
            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            # 把该样本分词结果追加到列表
            tokenized_batch.append(batch_element)

        # return collated batch
        # 把整批分词结果交给 collate 补齐后返回
        return self.collate(tokenized_batch)
    
# 定义工具函数 pad_to_length：把张量沿指定维度补齐到给定长度
def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    # 若该维度已不小于目标长度，无需补齐
    if tensor.size(dim) >= length:
        # 直接原样返回张量
        return tensor
    # 否则进入补齐分支
    else:
        # 复制张量形状为列表，用于描述待拼接填充块的形状
        pad_size = list(tensor.shape)
        # 把目标维度的大小改为差值（length 减当前长度），即需要补的元素数
        pad_size[dim] = length - tensor.size(dim)
        # 用 torch.cat 沿目标维度拼接原张量与填充块
        return torch.cat(
            # 开始拼接列表的第一个元素
            [
                # 第一个元素是原张量
                tensor,
                # 第二个元素是按 pad_size 生成、与张量同 dtype 同设备的全 pad_value 张量
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            # 结束拼接列表
            ],
            # 指定沿 dim 维度拼接
            dim=dim,
        # 结束 torch.cat 调用并返回补齐后的张量
        )