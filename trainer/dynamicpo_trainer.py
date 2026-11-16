# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/lifcenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 文件级说明：本文件是 DynamicPO 论文最核心的训练器，自定义继承 transformers.Trainer 的 DPOTrainer，实现偏好优化训练逻辑。
# 整体位于 sft.py 监督微调之后、inference.py 推理之前，负责 DMPO 与 DynamicPO_DMPO 两种损失推导、动态 beta 调整与 KMeans 边界负样本筛选。
# 导入 warnings 模块，用于按类别屏蔽第三方库告警，配合第 36 行的 filterwarnings 生效。
import warnings
# 导入 defaultdict，用于构建 self._stored_metrics 这类嵌套默认字典，自动为不存在的键创建 list 以累积各 step 的指标。
from collections import defaultdict
# 从 typing 导入 Any/Callable/Dict/List/Literal/Optional/Tuple/Union，为函数签名与张量容器提供类型标注，其中 Literal 限定 train_eval 只能取 train 或 eval。
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
# 导入 importlib，供 is_peft_available 用 find_spec 探测 peft 是否安装，决定是否允许启用 PEFT/LoRA 包装分支。
import importlib
# 导入 numpy 并别名 np，用于把张量转成数组做 KMeans 聚类、求均值与 argmax 等数值运算。
import numpy as np
# 导入 math 标准库提供通用数学函数；本文件主要数值计算在 torch 上完成，此处保留以备扩展。
import math
# 导入 torch 主包，提供张量、logsigmoid、tanh、stack、no_grad 等构建 DPO 损失与指标统计的核心运算。
import torch
# 导入 torch.nn 并别名 nn，用于类型标注 model: nn.Module 及引用神经网络模块基类。
import torch.nn as nn
# 导入函数式接口 F，核心用到 F.logsigmoid 计算 DMPO/DynamicPO 的负对数 sigmoid 偏好损失。
import torch.nn.functional as F
# 导入 HuggingFace datasets 的 Dataset 类型，用于标注 train_dataset 与 eval_dataset 参数。
from datasets import Dataset
# 从 transformers 导入 DataCollator、PreTrainedModel、PreTrainedTokenizerBase、Trainer、TrainingArguments，作为自定义 DPOTrainer 的基类与参数类型。
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
# 导入 TrainerCallback 类型，用于 callbacks 参数的类型标注，支持训练过程中注入回调。
from transformers.trainer_callback import TrainerCallback
# 导入分布式通信包 dist，供多卡训练时进程间同步使用；本文件后续以 accelerator 为主。
import torch.distributed as dist
# 从同目录 utils 导入 DPODataCollatorWithPadding（把 prompt、chosen 与多个 rejected 分词并 padding）与 pad_to_length（把张量补齐到定长）。
from .utils import DPODataCollatorWithPadding, pad_to_length
# 导入 sklearn 的 KMeans，用于 _cluster_high_reward_indices 把负样本 reward 聚类、挑出高分段边界负样本。
from sklearn.cluster import KMeans
# 导入 omegaconf 的 DictConfig 与 OmegaConf，用于读取层级配置；本训练器以 Trainer 原生参数为主，保留配置能力。
from omegaconf import DictConfig,OmegaConf

# 再次导入 dist（原文件重复保留，与第 27 行一致），不改动原有导入结构。
import torch.distributed as dist
# 导入 os，用于设置环境变量 PYTHONWARNINGS 控制解释器层面的告警行为。
import os
# 再次导入 warnings（原文件重复保留），保证下方 filterwarnings 调用可用。
import warnings
# 把环境变量 PYTHONWARNINGS 设为 ignore，从解释器层面全局屏蔽告警，避免训练日志被警告刷屏。
os.environ["PYTHONWARNINGS"] = "ignore"
# 调用 filterwarnings(ignore) 二次屏蔽所有警告，与上一行配合保证过长或弃用告警不干扰指标输出。
warnings.filterwarnings("ignore")

# 定义全局常量 FIXED_MARGIN=6.0，作为 _build_dynamic_beta_records 里动态 beta 公式的 margin 默认值，衡量正样本到边界、边界到可辨别负样本两段间距的期望差。
FIXED_MARGIN = 6.0

# 定义探测函数 is_peft_available，返回布尔值，供 __init__ 判断是否允许走 PEFT/LoRA 包装分支。
def is_peft_available():
    # 用 find_spec 查找 peft 包，找不到即返回 None 表示未安装，据此决定能否对模型做 get_peft_model 包装。
    return importlib.util.find_spec("peft") is not None

# 若检测到 peft 已安装，才在模块级导入其专用函数，避免在未安装环境里直接触发 ImportError。
if is_peft_available():
    # 导入 get_peft_model（按 peft_config 注入 LoRA 适配器）与 prepare_model_for_kbit_training（为 4/8bit 量化模型训练做前置准备）。
    from peft import get_peft_model, prepare_model_for_kbit_training


# 定义 _group_batch_values：把按字段名索引的整批张量字典重排成按样本下标索引的各字段值列表字典，便于按单条样本遍历多个负样本。
def _group_batch_values(values_dict: Dict[str, torch.FloatTensor]) -> Dict[int, List[torch.Tensor]]:
    # 初始化结果字典，键为样本下标 i，值为该样本在每个字段上的张量列表。
    grouped_values: Dict[int, List[torch.Tensor]] = {}
    # 取字典里第一个字段名，仅用于读取整批的 batch_size，避免依赖某个固定字段名。
    first_key = next(iter(values_dict))
    # 用首个字段对应张量的长度得到批大小，作为后续按样本下标遍历的次数。
    batch_size = len(values_dict[first_key])
    # 遍历每个字段及其整批张量，字段名用不到故以 _ 丢弃，只保留 value。
    for _, value in values_dict.items():
        # 对批内每个样本下标 i 遍历，把各字段在该样本的值聚合到一起。
        for i in range(batch_size):
            # 把当前字段在第 i 个样本的值追加到该样本列表；setdefault 保证下标首次出现时自动建立空列表。
            grouped_values.setdefault(i, []).append(value[i])
    # 返回按样本分组后的字典，供 _select_boundary_critical_samples 与 _to_python_nested_list 使用。
    return grouped_values


# 定义 _to_python_nested_list：把按样本分组的张量值转成纯 Python float 的嵌套列表，便于写入 metrics 日志并脱离计算图。
def _to_python_nested_list(values: Dict[int, List[torch.Tensor]]) -> List[List[float]]:
    # 初始化嵌套列表结果，外层对应样本，内层对应该样本的多个负样本标量。
    nested_values: List[List[float]] = []
    # 按样本数量遍历每个下标 i，逐条转换。
    for i in range(len(values)):
        # 对第 i 个样本的每个张量先 detach 断开梯度、移到 CPU、取标量再转 float，组成该行列表后追加。
        nested_values.append([float(tensor.detach().cpu().item()) for tensor in values[i]])
    # 返回可直接写入日志字典的纯数值嵌套列表。
    return nested_values


# 定义 _cluster_high_reward_indices：当没有负样本越过阈值时，用 KMeans 按 reward 聚类并挑出 reward 最高的簇作为边界关键负样本，默认最多分 3 簇。
def _cluster_high_reward_indices(rewards: List[torch.Tensor], num_clusters: int = 3) -> List[int]:
    # 把负样本 reward 张量逐个 detach、cpu、item 并转 float 后组成一维 numpy 数组，供 KMeans 与均值统计使用。
    reward_array = np.array([float(tensor.detach().cpu().item()) for tensor in rewards])
    # 实际簇数取 min(num_clusters, 样本数)，避免负样本数少于 3 时 KMeans 因簇数超过样本数报错。
    cluster_num = min(num_clusters, len(reward_array))
    # 若样本数不足以分簇（小于等于 1），无需聚类。
    if cluster_num <= 1:
        # 直接返回全部下标，把所有负样本都当作边界关键样本，保证下游筛选结果不为空。
        return list(range(len(reward_array)))

    # 用 KMeans 把 reward reshape 成 (n,1) 后聚类，random_state=42 固定随机种子保证同一批数据聚类结果可复现。
    kmeans = KMeans(n_clusters=cluster_num, random_state=42).fit(reward_array.reshape(-1, 1))
    # 取出每个负样本所属簇的标签数组，用于按簇分组求均值。
    labels = kmeans.labels_
    # 对每个簇求其成员 reward 的均值，得到各簇的平均奖励水平，用于比较哪一簇最高。
    cluster_means = [np.mean(reward_array[labels == idx]) for idx in range(cluster_num)]
    # 取平均 reward 最大簇的标签，用 where 找出属于该簇的下标并转成 Python 列表，即高分段边界负样本。
    return np.where(labels == int(np.argmax(cluster_means)))[0].tolist()


# 定义 _select_boundary_critical_samples：动态负样本筛选核心，逐样本挑出边界关键负样本与可辨别负样本，并算出可辨别负样本的均值。
def _select_boundary_critical_samples(
    # 参数 target_values：按负样本字段分组的 logratio 张量（通常为 rejected_logratios），用于按筛选下标取出对应值。
    target_values: Dict[str, torch.FloatTensor],
    # 参数 threshold_values：每条样本的阈值张量（此处传入 policy_chosen_logps），负样本 logp 达到或超过它即视为难以区分的边界负样本。
    threshold_values: torch.FloatTensor,
    # 参数 policy_rejected_values：策略模型对各负样本的 logp 字典，用于与阈值比较及计算可辨别负样本均值。
    policy_rejected_values: Dict[str, torch.FloatTensor],
# 函数返回四元组：边界负样本的 target 列表、边界负样本的策略 logp 列表、每样本可辨别负样本均值、以及全部负样本 logp 的纯数值嵌套列表。
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], List[torch.Tensor], List[List[float]]]:
    # 把 target_values 按样本下标重排，得到每条样本在各负样本字段上的 logratio 列表。
    batch_target = _group_batch_values(target_values)
    # 把策略负样本 logp 同样按样本下标重排，供逐样本做阈值比较与均值计算。
    batch_policy_rejected = _group_batch_values(policy_rejected_values)
    # 把分组后的策略 logp 转成纯 Python 嵌套列表，作为 all_negative_reward 指标记录每条样本的全部负样本得分。
    batch_policy_rejected_list = _to_python_nested_list(batch_policy_rejected)

    # 初始化列表，保存每条样本被选中的边界负样本对应的 target（logratio）值。
    boundary_critical_sample_list: List[List[torch.Tensor]] = []
    # 初始化列表，保存每条样本被选中的边界负样本对应的策略 logp。
    policy_boundary_critical_list: List[List[torch.Tensor]] = []
    # 初始化列表，保存每条样本未被选为边界的可辨别负样本 logp 均值，作为动态 beta 的参照基线。
    policy_model_discriminative_mean_list: List[torch.Tensor] = []

    # 按样本下标逐条处理批内每个训练样本。
    for sample_idx in range(len(batch_policy_rejected)):
        # 取第 sample_idx 条样本在所有负样本字段上的 logratio 列表。
        sample_target_values = batch_target[sample_idx]
        # 取第 sample_idx 条样本在所有负样本上的策略 logp 列表。
        sample_policy_values = batch_policy_rejected[sample_idx]

        # 构造边界关键负样本下标列表：筛选策略 logp 达到或超过正样本阈值 threshold_values[sample_idx] 的负样本。
        boundary_critical_indices = [
            # 列表推导：遍历该样本每个负样本 logp，凡大于等于阈值（即模型已给高分、接近或超过正样本）的下标都入选，代表最难区分的边界负样本。
            idx for idx, value in enumerate(sample_policy_values) if value >= threshold_values[sample_idx]
        # 结束列表推导，得到本样本全部边界负样本下标。
        ]
        # 若一个负样本都没越过阈值（说明负样本整体分数偏低），走退化处理。
        if len(boundary_critical_indices) < 1:
            # 改用 KMeans 聚类挑出 reward 最高的一簇作为边界负样本，保证筛选结果非空。
            boundary_critical_indices = _cluster_high_reward_indices(sample_policy_values)

        # 构造可辨别负样本下标列表：取所有未被选为边界的负样本，即模型能明确给出较低分的简单负样本。
        model_discriminative_indices = [
            # 列表推导：遍历全部负样本下标，排除已在边界集合中的，剩下的即模型可轻松区分的负样本。
            idx for idx in range(len(sample_policy_values)) if idx not in boundary_critical_indices
        # 结束列表推导，得到本样本可辨别负样本下标。
        ]
        # 若所有负样本都被判为边界（没有可辨别负样本），走退化处理。
        if len(model_discriminative_indices) < 1:
            # 把边界负样本集合复用为可辨别集合，避免后续求均值时除以零。
            model_discriminative_indices = boundary_critical_indices

        # 按边界下标取出对应的 target（logratio）值，作为该样本参与 DynamicPO 损失的难负样本项。
        boundary_critical_sample_list.append([sample_target_values[idx] for idx in boundary_critical_indices])
        # 按边界下标取出对应的策略 logp，供构建动态 beta 时衡量正样本到边界的间距。
        policy_boundary_critical_list.append([sample_policy_values[idx] for idx in boundary_critical_indices])
        # 计算该样本可辨别负样本 logp 的均值并追加，作为边界以下分数段的参照基线。
        policy_model_discriminative_mean_list.append(
            # 把可辨别负样本 logp 求和后除以其个数得到平均值；分母由前面的非空保证不会为零。
            sum(sample_policy_values[idx] for idx in model_discriminative_indices) / len(model_discriminative_indices)
        # 结束 append 调用，完成该样本可辨别负样本均值的记录。
        )

    # 以元组形式返回四类筛选结果，供 preference_loss 的 DynamicPO_DMPO 分支使用。
    return (
        # 第 1 项：每条样本边界负样本的 target（logratio）列表，用于按动态 beta 加权算损失。
        boundary_critical_sample_list,
        # 第 2 项：每条样本边界负样本的策略 logp 列表，用于计算 delta_pos2boundary。
        policy_boundary_critical_list,
        # 第 3 项：每条样本可辨别负样本的 logp 均值，用于计算 delta_boundary2discriminative。
        policy_model_discriminative_mean_list,
        # 第 4 项：全部负样本 logp 的纯数值嵌套列表，写入 all_negative_reward 指标。
        batch_policy_rejected_list,
    # 结束返回元组的构造。
    )


# 定义 _build_dynamic_beta_records：DynamicPO 的核心，为每个边界负样本依据两段 logp 间距动态缩放 beta，使难负样本获得更大的梯度权重。
def _build_dynamic_beta_records(
    # 参数 policy_chosen_logps：策略模型对正样本的 logp，作为正样本到边界间距的起点。
    policy_chosen_logps: torch.FloatTensor,
    # 参数 policy_boundary_critical_list：每样本边界负样本的策略 logp 列表，逐条参与 beta 计算。
    policy_boundary_critical_list: List[List[torch.Tensor]],
    # 参数 policy_model_discriminative_mean_list：每样本可辨别负样本 logp 均值，作为边界到可辨别间距的终点。
    policy_model_discriminative_mean_list: List[torch.Tensor],
    # 参数 beta：DPO 温度系数基准值，动态 beta 在此基础上按 (1+delta_beta) 缩放。
    beta: float,
    # 参数 margin：两段间距差的期望目标，默认取全局 FIXED_MARGIN=6.0，控制 beta 调整的方向与幅度。
    margin: float = FIXED_MARGIN,
# 返回三组逐样本、逐负样本的记录：动态 beta 值、正样本到边界间距、边界到可辨别间距，均用于损失计算与指标日志。
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    # 初始化外层列表，按样本保存其内部每个边界负样本对应的动态 beta 值。
    sample_level_beta_record: List[List[float]] = []
    # 初始化记录，按样本保存每个边界负样本的 delta_pos2boundary（正样本减边界 logp）数值。
    delta_pos2boundary_record: List[List[float]] = []
    # 初始化记录，按样本保存每个边界负样本的 delta_boundary2discriminative（边界减可辨别均值 logp）数值。
    delta_boundary2discriminative_record: List[List[float]] = []

    # 按批大小逐样本遍历，policy_chosen_logps 第 0 维即 batch_size。
    for i in range(policy_chosen_logps.size(0)):
        # 为当前样本初始化动态 beta 列表，每个边界负样本一个值。
        beta_record = []
        # 为当前样本初始化正样本到边界间距列表。
        pos2boundary_record = []
        # 为当前样本初始化边界到可辨别间距列表。
        boundary2discriminative_record = []

        # 遍历第 i 个样本的每一个边界负样本 logp，逐一计算其专属动态 beta。
        for boundary_critical_logp in policy_boundary_critical_list[i]:
            # 以全局 beta 为基准值起步，后续按该负样本的间距关系缩放。
            beta_value = beta
            # 计算正样本与当前边界负样本的 logp 差：值越小说明该负样本越逼近正样本、越难区分。
            delta_pos2boundary = policy_chosen_logps[i] - boundary_critical_logp
            # 计算边界负样本与可辨别负样本均值的 logp 差，衡量边界负样本高出易区分负样本的程度。
            delta_boundary2discriminative = (
                # 用当前边界负样本 logp 减去该样本可辨别负样本 logp 均值，得到第二段间距。
                boundary_critical_logp - policy_model_discriminative_mean_list[i]
            # 结束括号表达式，得到 delta_boundary2discriminative 张量。
            )

            # 把正样本到边界间距脱离计算图、取标量转 float 后记录，供后续指标日志使用。
            pos2boundary_record.append(float(delta_pos2boundary.detach().cpu().item()))
            # 把边界到可辨别间距同样转成 Python float 记录，便于观察两段间距的分布。
            boundary2discriminative_record.append(float(delta_boundary2discriminative.detach().cpu().item()))

            # 两段间距之差再减去 margin：衡量当前负样本相对期望间隔的偏离，正值表示正样本到边界间距相对过大、需增强该难负样本权重。
            delta_diff = delta_pos2boundary - delta_boundary2discriminative - margin
            # 用两段间距的绝对值之和做归一化，把偏离量映射到与尺度无关的相对比例，避免 logp 量级影响调整幅度。
            delta_beta = delta_diff / (abs(delta_boundary2discriminative) + abs(delta_pos2boundary))
            # 经 tanh 压缩到 (-1,1) 再乘 0.5，把相对调整量限制在 (-0.5,0.5)，保证 beta 只在基准的 0.5 到 1.5 倍间平滑变化、防止梯度爆炸。
            delta_beta = torch.tanh(delta_beta) * 0.5

            # 把基准 beta 按 (1+delta_beta) 缩放，得到该边界负样本专属的动态 beta：难负样本对应更大 beta、更易被推开。
            beta_value = beta_value * (1 + delta_beta)
            # 把动态 beta 转成 Python float 存入当前样本的记录列表。
            beta_record.append(float(beta_value.detach().cpu().item()))

        # 把当前样本所有边界负样本的动态 beta 列表加入样本级记录。
        sample_level_beta_record.append(beta_record)
        # 把当前样本的正样本到边界间距列表加入样本级记录。
        delta_pos2boundary_record.append(pos2boundary_record)
        # 把当前样本的边界到可辨别间距列表加入样本级记录。
        delta_boundary2discriminative_record.append(boundary2discriminative_record)

    # 返回三组样本级记录，供 preference_loss 按 [i][idx] 取用动态 beta 计算加权损失，并写入指标。
    return sample_level_beta_record, delta_pos2boundary_record, delta_boundary2discriminative_record



# 定义 preference_loss：偏好优化损失的总入口，按 filter_mode 分派到 DMPO 或 DynamicPO_DMPO，返回损失、奖励与动态 beta 记录。
def preference_loss(
                    # 参数 ref_model_enabled：是否启用参考模型；为 True 时 logratio 需减去 reference logp，为 False 时隐式参考模型对所有响应等概率、直接用策略 logp。
                    ref_model_enabled:bool,
                    # 参数 policy_chosen_logps：策略模型对正样本 chosen 的 logp 张量，形状 (batch_size,)。
                    policy_chosen_logps: torch.FloatTensor,
                    # 参数 policy_rejected_logps：策略模型对各负样本的 logp 字典（键为 rejected1 到 rejectedN），支持每条样本多个负样本。
                    policy_rejected_logps: torch.FloatTensor,
                    # 参数 reference_chosen_logps：参考模型对正样本的 logp，用于构造 chosen 的 logratio。
                    reference_chosen_logps: torch.FloatTensor,
                    # 参数 reference_rejected_logps：参考模型对各负样本的 logp 字典，与 policy_rejected_logps 的键一一对应。
                    reference_rejected_logps: torch.FloatTensor,
                    # 参数 filter_mode：损失模式开关，取 DMPO（多负样本平均）或 DynamicPO_DMPO（动态 beta 加边界负样本筛选）。
                    filter_mode: str,
                    # 参数 beta：DPO 温度系数；签名返回声明为三元组张量，实际各分支返回更丰富的元组（损失、奖励、beta 记录等），由调用方按多返回值解包。
                    beta: float) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    # 函数文档字符串开头，说明本函数针对一批策略模型与参考模型的 log 概率计算 DPO 损失。
    # Args 之 policy_chosen_logps：策略模型对 chosen 响应的 log 概率，形状 (batch_size,)。
    # Args 之 policy_rejected_logps：策略模型对 rejected 响应的 log 概率，形状 (batch_size,)。
    # Args 之 reference_chosen_logps 与 reference_rejected_logps：参考模型对 chosen 与 rejected 的 log 概率，均为 (batch_size,)。
    # Args 之 beta：DPO 损失温度参数，典型取 0.1 到 0.5；beta 趋于 0 时等价于忽略参考模型。
    # Args 之 reference_free：为 True 时忽略传入的参考模型，隐含假设参考模型对所有响应赋予相同概率。
    # Returns 说明：返回 (losses, chosen_rewards, rejected_rewards) 三元组，losses 为逐样本 DPO 损失。
    # Returns 说明：chosen_rewards 与 rejected_rewards 分别为正样本与被拒样本的隐式奖励值。
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """


    # 计算 chosen 的 logratio：启用参考模型时为 policy 减 reference，否则隐式等概率参考下直接等于策略 logp。
    chosen_logratios = policy_chosen_logps - reference_chosen_logps if ref_model_enabled else policy_chosen_logps
    # 计算 chosen 的隐式奖励：beta 乘 logratio 并 detach 断开梯度，仅作指标记录不参与反传；无参考模型时直接乘策略 logp。
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach() if ref_model_enabled else beta*policy_chosen_logps.detach()
   
    # 初始化字典，存放每个负样本字段（rejected1 到 rejectedN）对应的 logratio 张量。
    rejected_logratios = {}
    # 遍历每个负样本字段键，逐一计算其 logratio。
    for key in policy_rejected_logps:
        # 对第 key 个负样本：启用参考模型时取策略与参考 logp 之差，否则直接取策略 logp，存入 rejected_logratios 字典。
        rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key] if ref_model_enabled else policy_rejected_logps[key]

    # 初始化字典，存放每个负样本字段对应的隐式奖励张量。
    rejected_rewards = {}
    # 遍历每个负样本字段键，计算其隐式奖励。
    for key in policy_rejected_logps:
        # 第 key 个负样本的隐式奖励：beta 乘 logratio（或策略 logp）后 detach，仅用于指标统计。
        rejected_rewards[key] = beta * (policy_rejected_logps[key] - reference_rejected_logps[key]).detach() if ref_model_enabled else beta*policy_rejected_logps[key].detach()



   
   

    # DMPO 分支：多负样本 DPO，把所有负样本的 logratio 取平均后与 chosen 做标准的负 logsigmoid 偏好损失。
    if filter_mode== 'DMPO':
        # K 为负样本个数（rejected1 到 rejectedN 的数量），用作平均的分母，对应入口脚本的 neg_num。
        K=len(rejected_logratios.keys())
        # DMPO 损失：-logsigmoid(beta*chosen_logratios - sum(beta*rejected_logratios[key])/K)，即先把 K 个负样本的 beta 加权 logratio 取平均，再与正样本做 DPO 对比。
        losses=-F.logsigmoid(beta*chosen_logratios-sum(beta*rejected_logratios[key] for key in rejected_logratios)/K)
        # 返回 (losses,None) 作为损失二元组、边界与全负样本奖励占位 None、正负奖励，以及 (beta,beta,beta) 三个占位值供调用方解包为 beta_used 与两段 delta 记录。
        return (losses,None),None,None,chosen_rewards,rejected_rewards,(beta,beta,beta)
    
    
  
  
    # DynamicPO_DMPO 分支：先筛选边界关键负样本、再逐样本逐负样本算动态 beta，最后用动态 beta 加权的 DMPO 损失。
    elif filter_mode == "DynamicPO_DMPO":
        # 开始解包 _select_boundary_critical_samples 的四元返回值，用多行括号提高可读性。
        (
            # 接收第 1 项：每样本边界负样本的 target（logratio）列表，作为动态加权损失里的负样本项。
            boundary_critical_sample_list,
            # 接收第 2 项：每样本边界负样本的策略 logp 列表，用于构建动态 beta。
            policy_boundary_critical_list,
            # 接收第 3 项：每样本可辨别负样本 logp 均值。
            policy_model_discriminative_mean_list,
            # 接收第 4 项：全部负样本 logp 的纯数值列表，作为 all_negative_reward 指标。
            batch_policy_rejected_list,
        # 调用边界负样本筛选函数，传入三项参数并返回上述四个结果。
        ) = _select_boundary_critical_samples(
            # 把各负样本的 logratio 作为 target_values，筛选后按边界下标取出参与损失的值。
            target_values=rejected_logratios,
            # 以策略正样本 logp 为阈值：负样本 logp 达到或超过它即视为难区分的边界负样本。
            threshold_values=policy_chosen_logps,
            # 传入策略模型对各负样本的 logp，用于阈值比较与可辨别负样本均值计算。
            policy_rejected_values=policy_rejected_logps,
        # 结束 _select_boundary_critical_samples 调用。
        )

        # 开始解包 _build_dynamic_beta_records 的三元返回值。
        (
            # 接收第 1 项：逐样本、逐边界负样本的动态 beta 记录，作为损失加权系数。
            sample_level_beta_record,
            # 接收第 2 项：逐样本的正样本到边界间距记录，写入指标日志。
            delta_pos2boundary_record,
            # 接收第 3 项：逐样本的边界到可辨别间距记录，写入指标日志。
            delta_boundary2discriminative_record,
        # 调用动态 beta 构建函数，基于筛选出的边界负样本计算每个负样本的专属 beta。
        ) = _build_dynamic_beta_records(
            # 传入策略正样本 logp，作为两段间距计算的起点。
            policy_chosen_logps=policy_chosen_logps,
            # 传入各样本边界负样本 logp，逐条计算 beta。
            policy_boundary_critical_list=policy_boundary_critical_list,
            # 传入各样本可辨别负样本均值，作为第二段间距的参照。
            policy_model_discriminative_mean_list=policy_model_discriminative_mean_list,
            # 传入基准 beta；margin 省略，由函数默认取 FIXED_MARGIN=6.0。
            beta=beta,
        # 结束 _build_dynamic_beta_records 调用。
        )

        # 初始化列表，逐样本收集 DynamicPO_DMPO 损失标量后再堆叠成批张量。
        batch_losses_list = []
        # 按 batch_size 逐样本计算损失，因每个样本的边界负样本个数与动态 beta 都不同，需单独处理。
        for i in range(chosen_logratios.size(0)):
            # k 为第 i 个样本被选中的边界负样本个数，作为该样本损失里各负样本项的平均分母。
            k = len(boundary_critical_sample_list[i])
            # 对第 i 个样本计算负 logsigmoid 形式的 DMPO 损失，内部按边界负样本求和并加动态 beta 权重。
            dmpo_loss_item = -F.logsigmoid(
                # 对该样本所有边界负样本的贡献求和，构成 logsigmoid 的输入。
                sum(
                    # 正样本项：用该边界负样本对应的动态 beta 乘以 chosen logratio 再除以 k，使多负样本时正样本贡献按负样本数均摊。
                    sample_level_beta_record[i][idx] * chosen_logratios[i] / k
                    # 负样本项：同一动态 beta 乘以该边界负样本 logratio 再除以 k 并取负，与正样本项配对形成逐负样本的 DPO 对比。
                    - sample_level_beta_record[i][idx] * neg_sample_logratio / k
                    # 生成器遍历第 i 个样本的每个边界负样本 logratio 及其下标 idx，用于索引对应的动态 beta。
                    for idx, neg_sample_logratio in enumerate(boundary_critical_sample_list[i])
                # 结束 sum 求和，得到该样本 logsigmoid 的加权输入。
                )
            # 结束负 logsigmoid 调用，得到第 i 个样本的标量损失。
            )
            # 把该样本损失标量加入列表，供后续堆叠为批损失。
            batch_losses_list.append(dmpo_loss_item)

        # 把逐样本损失标量堆叠成形状 (batch_size,) 的张量，供 Trainer 求均值反传。
        losses = torch.stack(batch_losses_list)

        # 返回 DynamicPO_DMPO 分支的多项结果，与 DMPO 分支保持相同的解包结构以便调用方统一处理。
        return (
            # 损失二元组：(逐样本损失张量, None)，None 为 margin_record 占位以保持结构一致。
            (losses, None),
            # 边界负样本的策略 logp 列表，作为 boundary_critical_reward 指标记录。
            policy_boundary_critical_list,
            # 全部负样本 logp 列表，作为 all_negative_reward 指标记录。
            batch_policy_rejected_list,
            # 正样本隐式奖励张量，供指标统计。
            chosen_rewards,
            # 各负样本隐式奖励字典，供指标统计。
            rejected_rewards,
            # 三元组：动态 beta 记录与两段间距记录，供调用方解包为 beta_used、delta_pos2boundary_record、delta_boundary2discriminative_record。
            (sample_level_beta_record, delta_pos2boundary_record, delta_boundary2discriminative_record),
        # 结束返回元组的构造。
        )


   
    

# 定义 DPOTrainer，继承 transformers.Trainer 并重写 compute_loss、prediction_step、log 等，注入 DMPO/DynamicPO 的偏好优化训练逻辑。
class DPOTrainer(Trainer):
    # 类文档字符串开头（r 前缀表示 raw string，避免反斜杠被转义），说明这是 DPOTrainer 的初始化参数说明。
    # Initialize DPOTrainer：概述本类负责初始化偏好优化训练器。
    # Args 之 model：待训练模型，推荐 AutoModelForSequenceClassification，实为带因果语言建模头的生成模型。
    # Args 之 ref_model：带 causal LM 头的参考模型，用于隐式奖励计算与 DPO 损失里的 logratio 基准。
    # Args 之 beta：DPO 损失系数，默认 0.1；beta 越大策略相对初始策略的偏离越小。
    # Args 之 args：transformers.TrainingArguments，承载批大小、学习率、日志与保存等训练配置。
    # Args 之 data_collator：为 None 时使用默认 DPODataCollatorWithPadding，把成对序列 pad 到批内最大长度。
    # Args 之 label_pad_token_id：标签填充值，默认 -100；使用默认 collator 时必填，用于 mask 掉 prompt 段。
    # Args 之 padding_value：输入与注意力掩码的填充值，默认 0；使用默认 collator 时必填。
    # Args 之 truncation_mode：截断方式 keep_end（保尾）或 keep_start（保头），默认 keep_end。
    # Args 之 train_dataset 与 eval_dataset：训练与评估用的 datasets.Dataset。
    # Args 之 tokenizer：分词器，使用默认 collator 时必填，负责编码 prompt 与各响应。
    # Args 之 model_init：可调用的模型初始化器，为 None 时用 Trainer 默认初始化。
    # Args 之 callbacks：TrainerCallback 列表，用于注入训练过程回调。
    # Args 之 optimizers：优化器与学习率调度器的二元组。
    # Args 之 preprocess_logits_for_metrics：计算指标前对 logits 预处理的函数。
    # Args 之 max_length 与 max_prompt_length：批内序列与 prompt 的最大长度，默认 None，使用默认 collator 时建议显式设置。
    # Args 之 peft_config：PEFT/LoRA 配置，传入时模型会被包装为 PEFT 模型以做参数高效微调。
    # 文档字符串结尾：以上参数共同决定 DPOTrainer 的数据整理、参考模型与偏好损失行为。
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
    """

    # 定义 DPOTrainer 构造函数，接收模型、参考模型、beta、损失模式及 Trainer 原生参数并完成组装。
    def __init__(
        # 实例自身引用，作为所有成员属性与父类初始化的挂载点。
        self,
        # 待训练的策略模型（PEFT/LoRA 包装后的基座模型），默认 None，由入口 DynamicPO.py 传入。
        model: Union[PreTrainedModel, nn.Module] = None,
        # 参考模型，默认 None；为 None 时走 reference_free 的隐式等概率参考，对应入口 loss_type 取 wo_ref 的情形。
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        # DPO 温度系数 beta，默认 1.0，可由入口命令行覆盖；越大对概率比差异越敏感。
        beta: float = 1.0,
        # 损失模式字符串，默认空；由脚本传 DMPO 或 DynamicPO_DMPO 决定 preference_loss 走哪个分支。
        filter_mode:str='',
        # Trainer 训练配置对象，承载批大小、梯度累积、bf16、保存策略等。
        args: TrainingArguments = None,
        # 数据整理器，默认 None 时构造函数自动创建 DPODataCollatorWithPadding。
        data_collator: Optional[DataCollator] = None,
        # 标签填充 id，默认 -100，使 prompt 部分 token 不计入 logp 损失。
        label_pad_token_id: int = -100,
        # 输入与注意力掩码的填充值，默认 0，与 tokenizer.pad_token_id 对齐。
        padding_value: int = 0,
        # 超长序列截断方式，默认 keep_end 保留尾部（更靠近回答的 token）。
        truncation_mode: str = "keep_end",
        # 训练数据集，默认 None，由入口传入含 prompt、chosen 与多个 rejected 的样本。
        train_dataset: Optional[Dataset] = None,
        # 评估数据集（可为单个 Dataset 或按名索引的字典），默认 None 表示不评估。
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # 分词器，默认 None；使用默认 collator 时必填，负责编码与 padding。
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        # 模型初始化可调用对象，默认 None 沿用 Trainer 默认逻辑。
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        # 训练回调列表，默认 None，可注入自定义日志或早停逻辑。
        callbacks: Optional[List[TrainerCallback]] = None,
        # 优化器与学习率调度器二元组，默认值以跨行元组形式给出，两项均为 None 表示交给 Trainer 自动创建。
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            # 优化器默认值 None，交由父类 Trainer 依据 training_args 构建。
            None,
            # 学习率调度器默认值 None，同样由父类 Trainer 自动构建。
            None,
        # 结束 optimizers 默认元组的定义。
        ),
        # 计算指标前对 logits 预处理的函数，默认 None 表示不做预处理。
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        # 批内序列最大长度，默认 None；未设置且用默认 collator 时构造函数会兜底为 512。
        max_length: Optional[int] = None,
        # prompt 最大长度，默认 None；未设置且用默认 collator 时会兜底为 128。
        max_prompt_length: Optional[int] = None,
        # PEFT/LoRA 配置字典，默认 None；非 None 时会把模型包装为 PEFT 模型。
        peft_config: Optional[Dict] = None,
    # 结束构造函数参数列表。
    ):
        # 若未安装 peft 却传入了 peft_config，无法走 PEFT 分支，需立即报错。
        if not is_peft_available() and peft_config is not None:
            # 抛出 ValueError 阻止继续初始化，给出明确的依赖安装提示。
            raise ValueError(
                # 报错信息：说明 peft 未安装却传入了 peft_config，要求安装 peft 后才能使用 PEFT 模型。
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            # 结束 raise ValueError 调用。
            )
        # 若 peft 可用且提供了 peft_config，则对模型做 PEFT/LoRA 包装。
        elif is_peft_available() and peft_config is not None:
            # 检测模型是否以 8bit 或 4bit 量化加载，量化模型训练前需先做适配准备。
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                # 对量化模型做 kbit 训练前置处理（如类型转换、梯度检查点、冻结等），保证 LoRA 能稳定训练。
                model = prepare_model_for_kbit_training(model)
            # 按 peft_config 给模型注入 LoRA 适配器并包装为 PeftModel，实现参数高效微调。
            model = get_peft_model(model, peft_config)

        # 若调用方未提供数据整理器，则进入默认 DPODataCollatorWithPadding 的构建分支。
        if data_collator is None:
            # 默认 collator 依赖 tokenizer 做编码，缺失则无法工作，需先校验。
            if tokenizer is None:
                # 抛出 ValueError，拒绝在没有 tokenizer 的情况下使用默认 collator。
                raise ValueError(
                    # 报错信息：使用默认 DPODataCollatorWithPadding 时必须提供 max_length 或 tokenizer。
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                # 结束 raise ValueError 调用。
                )
            # 若未显式设置最大序列长度，给出告警并采用兜底默认值。
            if max_length is None:
                # 发出 UserWarning 提醒调用方应显式设置 max_length。
                warnings.warn(
                    # 告警信息前半：使用 DPODataCollatorWithPadding 时应在 DPOTrainer 初始化里设置 max_length。
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    # 告警信息后半（与上一行隐式拼接）：当前兜底为 512，建议今后自行显式指定。
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    # 指定告警类别为 UserWarning，与相邻字符串参数共同构成 warn 的调用参数。
                    UserWarning,
                # 结束 warnings.warn 调用。
                )
                # 把 max_length 兜底设为 512，保证默认 collator 有明确的最大序列长度。
                max_length = 512
            # 若未显式设置 prompt 最大长度，同样告警并兜底。
            if max_prompt_length is None:
                # 发出 UserWarning 提醒调用方应显式设置 max_prompt_length。
                warnings.warn(
                    # 告警信息前半：使用默认 collator 时应设置 max_prompt_length。
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    # 告警信息后半：当前兜底为 128，建议今后自行显式指定。
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    # 指定告警类别为 UserWarning。
                    UserWarning,
                # 结束 warnings.warn 调用。
                )
                # 把 max_prompt_length 兜底设为 128，限制 prompt 的编码长度。
                max_prompt_length = 128

            # 用各项兜底或传入参数实例化默认数据整理器，负责把 prompt、chosen、多个 rejected 分词并 padding。
            data_collator = DPODataCollatorWithPadding(
                # 第 1 个参数：分词器，供 collator 编码文本与取 pad_token_id。
                tokenizer,
                # 最大序列长度，超长时按 truncation_mode 截断。
                max_length=max_length,
                # prompt 最大长度，超长优先截断 prompt。
                max_prompt_length=max_prompt_length,
                # 标签填充 id，用于把 prompt 段标签置为 -100 以 mask 掉其损失。
                label_pad_token_id=label_pad_token_id,
                # 输入与注意力掩码的填充值。
                padding_value=padding_value,
                # 截断方向 keep_end 或 keep_start，决定超长时保留哪一端。
                truncation_mode=truncation_mode,
            # 结束 DPODataCollatorWithPadding 实例化。
            )

            # 若 TrainingArguments 仍开启 remove_unused_columns，会丢弃自定义 collator 需要的 rejected 字段，需纠正。
            if args.remove_unused_columns:
                # 强制关闭该选项，保留 prompt、chosen、多个 rejected 等自定义列供 collator 使用。
                args.remove_unused_columns = False
                # warn users
                # 发出告警，告知调用方该选项已被自动改写。
                warnings.warn(
                    # 告警信息前半：使用默认 collator 时应把 TrainingArguments 的 remove_unused_columns 设为 False。
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    # 告警信息后半：已代为设置，建议今后自行显式配置。
                    " we have set it for you, but you should do it yourself in the future.",
                    # 指定告警类别为 UserWarning。
                    UserWarning,
                # 结束 warnings.warn 调用。
                )

            # 标记当前使用默认 DPO collator，compute_loss 与 prediction_step 据此确认数据格式兼容。
            self.use_dpo_data_collator = True
        # 若调用方自带了 data_collator，则跳过上述默认构建。
        else:
            # 标记为非默认 collator，后续 compute_loss 与 prediction_step 会发出兼容性告警。
            self.use_dpo_data_collator = False

        # 把标签填充 id 存为实例属性，供 _get_batch_logps 构造 loss_mask 使用。
        self.label_pad_token_id = label_pad_token_id
        # 把输入填充值存为实例属性，供 concatenated_inputs 补齐各序列。
        self.padding_value = padding_value

        # 保存 DPO 温度系数 beta，供 preference_loss 与动态 beta 基准使用。
        self.beta = beta
        # 保存参考模型（可能为 None），供 concatenated_forward 计算参考 logp。
        self.ref_model = ref_model
        # 把可能已 PEFT 包装的模型另存为 self.policy，便于与参考模型区分。
        self.policy=model
 
        # 保存损失模式字符串，_get_batch_metrics 调用 preference_loss 时据此分派分支。
        self.filter_mode=filter_mode


    
        # 在 CUDA 上初始化边界到可辨别间距的均值缓存（长度 1 的零张量），用于指标记录。
        self.delta_boundary2discriminative_mean = torch.zeros(1, device='cuda')
        # 在 CUDA 上初始化该间距的标准差缓存，配合均值记录分布。
        self.delta_boundary2discriminative_std = torch.zeros(1, device='cuda')

        # 在 CUDA 上初始化正样本到边界间距的均值缓存。
        self.delta_pos2boundary_mean = torch.zeros(1, device='cuda')
        # 在 CUDA 上初始化该间距的标准差缓存。
        self.delta_pos2boundary_std = torch.zeros(1, device='cuda')

        # 创建嵌套默认字典：外层按 train 或 eval 分组，内层按指标名累积为 list，供 store_metrics 追加、log 求平均。
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # 调用父类 Trainer 构造函数，按位置参数传入各项配置完成底层训练器初始化。
        super().__init__(
            # 第 1 个位置参数：策略模型，交给 Trainer 管理。
            model,
            # TrainingArguments 训练配置。
            args,
            # 已构建（或调用方传入）的数据整理器。
            data_collator,
            # 训练数据集。
            train_dataset,
            # 评估数据集。
            eval_dataset,
            # 分词器。
            tokenizer,
            # 模型初始化器。
            model_init,
            # 对应 compute_metrics 的位置参数，传 None 表示不使用额外指标函数。
            None,
            # 训练回调列表。
            callbacks,
            # 优化器与调度器二元组。
            optimizers,
            # logits 预处理函数，作为父类构造的最后一个位置参数。
            preprocess_logits_for_metrics,
        # 结束 super().__init__ 调用。
        )

        # Since we inherit from trainer we always have access to an accelerator
        # 检测父类 Trainer 是否提供 accelerator（分布式与混合精度加速器），以决定能否准备参考模型。
        if hasattr(self, "accelerator"):
            # 若参考模型存在，用 accelerator.prepare_model 以 evaluation_mode=True 包装（只推理、不参与优化），否则保持 None。
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True) if self.ref_model is not None else None
        # 若当前 transformers 版本没有 accelerator 属性，则无法继续。
        else:
            # 抛出 AttributeError，提示升级 transformers。
            raise AttributeError(
                # 报错信息：Trainer 缺少 accelerator 对象，建议升级 transformers 版本。
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            # 结束 raise AttributeError 调用。
            )

    # 定义 concatenated_inputs：把 chosen 与多个 rejected 的输入张量在 batch 维拼接成一份，供单次前向计算所有序列的 logits。
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        # 函数文档字符串：说明本函数把 chosen 与 rejected 的输入拼接为单个张量。
        # Args 之 batch：须含 chosen_input_ids 与 rejected_input_ids 等键，各张量形状 (batch_size, sequence_length)。
        # Returns：结果字典里以 concatenated 为前缀的键保存拼接后的输入，如 concatenated_input_ids。
        # 补充：把所有响应拼在一起可只做一次前向、避免多次 forward，对 FSDP 更友好。
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        
        # 求所有 rejected 开头且以 _input_ids 结尾字段中的最大序列长度，作为拼接时各响应统一补齐的长度参考。
        rejected_max_len = max([batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
        # 取 chosen 长度与各 rejected 最大长度的较大者，作为整批所有序列补齐到的统一长度。
        max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
        # 初始化结果字典，存放以 concatenated 为前缀的拼接后张量。
        concatenated_batch = {}
        # 第一遍遍历批内所有键，处理 chosen 相关张量。
        for k in batch:
            # 仅处理以 chosen 开头且为张量的字段（如 input_ids、attention_mask、labels），跳过字符串等元数据。
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                # labels 字段用 -100 填充以 mask 掉损失，其余字段用 padding_value 即 0 填充。
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                # 把键名里的 chosen 替换为 concatenated，得到统一命名（如 concatenated_input_ids）。
                concatenated_key = k.replace("chosen", "concatenated")
                # 把 chosen 张量补齐到 max_length 后存入结果字典，作为拼接张量的前半部分。
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        # 第二遍遍历批内所有键，处理各 rejected 张量并按顺序追加。
        for k in batch:
            # 仅处理 rejected 前缀的张量字段，逐一把每个负样本拼到已有张量后面。
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                # 与 chosen 一致：labels 用 -100、其余用 0 作为填充值。
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                # concatenated_key = k.replace("rejected", "concatenated")
                # 取键名下划线前的第一段（如 rejected1），用于计算去掉前缀后的后缀长度。
                prefix = k.split("_")[0]
                # 用 concatenated 替换 rejectedN 前缀，得到与 chosen 相同的目标键名（如 concatenated_input_ids）。
                concatenated_key = "concatenated" + k[len(prefix):]
                # 把当前 rejected 张量沿 batch 维拼接到已有的 concatenated 张量尾部。
                concatenated_batch[concatenated_key] = torch.cat(
                    # 以元组形式给出 torch.cat 待拼接的两个张量。
                    (
                        # 第 1 个张量：此前已累积的 chosen 加前面各 rejected 的拼接结果。
                        concatenated_batch[concatenated_key],
                        # 第 2 个张量：当前 rejected 字段补齐到 max_length 后的结果。
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    # 结束待拼接张量元组。
                    ),
                    # 指定沿第 0 维（batch 维）拼接，使所有响应排成更长的批次。
                    dim=0,
                # 结束 torch.cat 并把结果移到 accelerator 所在设备，保证与模型同卡。
                ).to(self.accelerator.device)
        # 返回拼接完成的字典，供 concatenated_forward 一次性喂给模型。
        return concatenated_batch


    # 定义 _get_batch_logps：由模型 logits 与 labels 计算每条序列的 token 级 log 概率（求和或平均）。
    def _get_batch_logps(
        # 实例引用，用于取 self.label_pad_token_id 构造损失掩码。
        self,
        # 模型输出 logits，形状 (batch_size, sequence_length, vocab_size)。
        logits: torch.FloatTensor,
        # 标签 token id，形状 (batch_size, sequence_length)，填充处为 label_pad_token_id。
        labels: torch.LongTensor,
        # 为 True 返回每个非 mask token 的平均 logp，为 False 返回求和；默认求和，与 DPO 惯用设定一致。
        average_log_prob: bool = False,
    # 返回形状 (batch_size,) 的每条序列 logp 张量。
    ) -> torch.FloatTensor:
        # 函数文档字符串：说明本函数计算给定 labels 在给定 logits 下的 log 概率。
        # Args 之 logits：模型未归一化输出，形状 (batch_size, sequence_length, vocab_size)。
        # Args 之 labels：目标 token，值等于 label_pad_token_id 的位置会被忽略，形状 (batch_size, sequence_length)。
        # Args 之 average_log_prob：为 True 返回非 mask token 的平均 logp，否则返回求和。
        # Returns：形状 (batch_size,) 的张量，含每条序列的平均或求和 log 概率。
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        # 校验 logits 的 batch 与 sequence_length 两维与 labels 形状一致，否则无法逐 token 对齐。
        if logits.shape[:-1] != labels.shape:
            # 形状不符时抛出 ValueError，给出明确的维度要求提示。
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # 把 labels 左移一位（去掉第 0 个 token）并 clone，得到每个位置要预测的下一 token；clone 避免改动原张量。
        labels = labels[:, 1:].clone()
        # 对应地截掉 logits 最后一个时间步，使第 t 个 logit 正好预测第 t+1 个 label，完成因果对齐。
        logits = logits[:, :-1, :]
        # 构造布尔掩码：非填充位置为 True，用于只对回答段（非 prompt）token 累加 logp。
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        # 把填充位置的 label id 临时置 0 作为 dummy token，仅为 gather 能取到合法索引，其损失稍后由 loss_mask 屏蔽。
        labels[labels == self.label_pad_token_id] = 0

        # 对 logits 在词表维取 log_softmax，再用 gather 按 labels 取出对应 token 的 logp，squeeze 回 (batch, seq) 的逐 token log 概率。
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # 若要求按 token 平均，走归一化分支。
        if average_log_prob:
            # 掩码后对 token 维求和再除以有效 token 数，得到每条序列的平均 logp。
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        # 否则走求和分支（默认），即 DPO 使用的序列级 log 概率。
        else:
            
            # 掩码后对 token 维求和，得到每条响应（chosen 或各 rejected）的序列 logp，供构造 logratio。
            return (per_token_logps * loss_mask).sum(-1)

    # 定义 concatenated_forward：把拼接后的批喂给指定模型做一次前向，再切分出 chosen 与各 rejected 的 logps 和 logits。
    def concatenated_forward(
        # 参数含实例自身、待前向的 model（策略或参考模型）以及原始 batch（提供 chosen 条数以定位切片）。
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    # 返回四元组：chosen logps、按 rejectedN 分组的 logps 字典、chosen logits、rejected logits 字典。
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        # 函数文档字符串：说明本函数在给定批上做前向，并把 chosen 与 rejected 输入拼接后一起送入模型。
        # 实现动机：拼接后只做一次前向而非两次，对 FSDP 训练更快、显存更省。
        # 文档字符串结尾：以上两句共同解释为何采用单批拼接前向的设计。
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # 先把 chosen 与多个 rejected 拼接、补齐成单一批字典，作为单次前向的输入。
        concatenated_batch = self.concatenated_inputs(batch)
        
        # 对拼接后的批做一次模型前向，得到所有序列统一的 logits。
        all_logits = model(
            # 传入拼接后的 input_ids 作为模型第 1 个位置参数。
            concatenated_batch["concatenated_input_ids"],
            # 传入对应的注意力掩码，标识哪些位置是真实 token、哪些是 padding。
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        # 取出前向输出的 logits 并转 float32，保证后续 log_softmax 与 gather 的数值精度与稳定性。
        ).logits.to(torch.float32)

        # 由全部 logits 计算每条序列（chosen 加各 rejected）的 token 求和 logp。
        all_logps = self._get_batch_logps(
            # 传入拼接前向得到的全部 logits。
            all_logits,
            # 传入拼接后的 labels，用 -100 掩码屏蔽 prompt 段。
            concatenated_batch["concatenated_labels"],
            # 显式指定按 token 求和（非平均），与 DPO 序列 logp 定义一致；行尾原有的 # 为原文件自带标记，保持不变。
            average_log_prob=False#
        # 结束 _get_batch_logps 调用。
        )
        # 按 chosen 样本条数切出前一段，得到各正样本的序列 logp。
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        # 把 chosen 条数记为步长 step，后续按 step 等距切分每个 rejectedN 对应的 logp 段。
        step = batch["chosen_input_ids"].shape[0]
        # 初始化字典，按 rejected1 到 rejectedN 存放各负样本的 logp 切片。
        rejected_logps = {}
        # 初始化计数器，按遍历到的 rejected 字段顺序编号。
        cnt = 0
        # 遍历批内所有键，找出各 rejected 开头以 _input_ids 结尾的字段以切分 logp。
        for key in batch:
            # 仅处理以 rejected 开头、以 _input_ids 结尾的键，确保每个负样本只被计一次。
            if key.startswith("rejected") and key.endswith("_input_ids"):
                # 每遇到一个负样本字段，编号加 1，与 collator 里的 rejectedN 命名对应。
                cnt += 1
                # 从拼接 logp 中按 step*cnt 到 step*(cnt+1) 切出第 cnt 个负样本段的 logp，存入字典。
                rejected_logps[f"rejected{cnt}"] = all_logps[step*cnt : step*(cnt+1)]

        # 同样按 chosen 条数切出正样本段的原始 logits，供指标或调试使用。
        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        # 初始化字典，按 rejectedN 存放各负样本段的 logits。
        rejected_logits = {}
        # 重置计数器，按 rejected 字段顺序重新编号。
        cnt = 0
        # 再次遍历批内键，切分各负样本的 logits。
        for key in batch:
            # 同样只匹配 rejected 开头以 _input_ids 结尾的键，保证与 logp 切分顺序一致。
            if key.startswith("rejected") and key.endswith("_input_ids"):
                # 编号加 1，对应当前第 cnt 个负样本。
                cnt += 1
                # 按 step 切出第 cnt 个负样本段的 logits 存入字典。
                rejected_logits[f"rejected{cnt}"] = all_logits[step*cnt : step*(cnt+1)]
        # 返回策略或参考模型在 chosen 与各 rejected 上的 logps 与 logits，供 _get_batch_metrics 组装损失函数参数。
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
    


 
    # 定义 _get_batch_metrics：组织单次前向、参考模型 logp、调用 preference_loss 并汇总训练或评估指标的核心流程。
    def _get_batch_metrics(
        # 实例引用，用于取 filter_mode、beta、ref_model 与各指标缓存。
        self,
        # 当前待前向的策略模型，由 compute_loss 或 prediction_step 传入。
        model,
        # 经 collator 整理的一个批，含 prompt、chosen 与多个 rejected 的张量或列表。
        batch: Dict[str, Union[List, torch.LongTensor]],
        # 标记本次是训练还是评估，决定指标键前缀 train_ 或 eval_，默认 train。
        train_eval: Literal["train", "eval"] = "train",
    # 结束参数列表。
    ):
        # 函数文档字符串：说明本函数为训练或测试的一个批计算 DPO 损失及各类监控指标。
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        # 初始化指标字典，收集奖励、准确率、logp、动态 beta 等序列供 Trainer 记录。
        metrics = {}

        # 开始解包 concatenated_forward 的策略模型四元返回值。
        (
            # 接收策略模型对正样本的序列 logp。
            policy_chosen_logps,
            # 接收策略模型对各负样本的 logp 字典。
            policy_rejected_logps,
            # 接收正样本段 logits（本函数后续主要用 logp，保留以对齐返回结构）。
            policy_chosen_logits,
            # 接收各负样本段 logits 字典。
            policy_rejected_logits,
        # 对策略模型做拼接单次前向，得到上述策略侧 logps 与 logits。
        ) = self.concatenated_forward(model, batch)


        # 在关闭梯度上下文里计算参考模型前向，参考模型不更新、也无需构建计算图，节省显存。
        with torch.no_grad():
            # 开始解包参考模型的四元返回值。
            (
                # 接收参考模型对正样本的 logp（无参考模型时为 None）。
                reference_chosen_logps,
                # 接收参考模型对各负样本的 logp 字典（无参考模型时为 None）。
                reference_rejected_logps,
                # 丢弃参考模型的 chosen logits（本流程不需要）。
                _,
                # 丢弃参考模型的 rejected logits。
                _,
            # 有参考模型时做拼接前向取参考 logp，否则整体给四个 None，走隐式等概率参考分支。
            ) = self.concatenated_forward(self.ref_model, batch) if self.ref_model is not None else (None, None, None, None)
        

        # 有参考模型时算正样本 logratio（策略减参考），否则置 None；此处仅作局部变量，实际 preference_loss 内部另行计算。
        chosen_logratios = policy_chosen_logps - reference_chosen_logps if self.ref_model is not None else None
        # 初始化字典，准备存放各负样本的 logratio。
        rejected_logratios = {}
        # 遍历每个负样本字段，逐条计算其 logratio。
        for key in policy_rejected_logps:
            # 有参考模型时取策略与参考 logp 之差存入字典，否则置 None。
            rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key] if self.ref_model is not None else None


        # 开始解包 preference_loss 的返回：先取 (losses, margin_record) 子元组及若干并列返回值。
        (
            # 接收逐样本的偏好损失张量（DMPO 或 DynamicPO_DMPO）。
            losses,
            # 接收 margin_record（各分支多为 None 占位），供 convert_tensors_to_list 记录。
            margin_record,
        # 继续解包并列项：边界负样本奖励、全部负样本奖励、正负隐式奖励，并开启承载 beta 与两段 delta 记录的子元组。
        ), boundary_critical_reward, all_negative_reward, chosen_rewards, rejected_rewards, (
            # 接收本次使用的 beta：DMPO 分支为标量，DynamicPO 分支为逐样本逐负样本列表。
            beta_used,
            # 接收正样本到边界间距记录（列表或张量）。
            delta_pos2boundary_record,
            # 接收边界到可辨别间距记录（列表或张量）。
            delta_boundary2discriminative_record,
        # 调用偏好损失总入口，按 self.filter_mode 分派具体损失计算。
        ) = preference_loss(
            # 以是否存在参考模型决定 logratio 是否减去参考 logp。
            ref_model_enabled=self.ref_model is not None,
            # 传入策略正样本 logp。
            policy_chosen_logps=policy_chosen_logps,
            # 传入策略各负样本 logp 字典。
            policy_rejected_logps=policy_rejected_logps,
            # 传入参考正样本 logp（可能为 None）。
            reference_chosen_logps=reference_chosen_logps,
            # 传入参考各负样本 logp 字典（可能为 None）。
            reference_rejected_logps=reference_rejected_logps,
            # 传入损失模式，决定走 DMPO 还是 DynamicPO_DMPO 分支。
            filter_mode=self.filter_mode,
            # 传入基准 DPO 温度系数 beta。
            beta=self.beta
        
        # 结束 preference_loss 调用。
        )



        
        # 初始化偏好准确率累加器，初始为 None 以便首次赋值。
        reward_accuracies = None
        # 遍历每个负样本字段，逐一比较正样本奖励是否高于该负样本。
        for key in rejected_rewards:
            # 第一个负样本时直接初始化准确率张量。
            if reward_accuracies is None:
                # 首个负样本：逐样本判断正样本奖励是否大于该负样本，转 float 得 0 或 1 的准确率。
                reward_accuracies = (chosen_rewards > rejected_rewards[key]).float()
            # 后续负样本：把判断结果累积相乘。
            else:
                # 与每个负样本的 0 或 1 判断相乘，只有正样本奖励高于全部负样本时结果才为 1，即多负样本联合准确率。
                reward_accuracies *= (chosen_rewards > rejected_rewards[key]).float()

        # 按 train 或 eval 决定指标键前缀，使训练与评估指标在日志中区分。
        prefix = "eval_" if train_eval == "eval" else "train_"
        # 记录正样本隐式奖励的批均值，观察策略对正样本的打分趋势。
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        # 遍历每个负样本字段，分别记录其奖励均值。
        for key in rejected_rewards:
            # 记录第 key 个负样本隐式奖励的批均值，观察各负样本被压低程度。
            metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()

        # 把多负样本联合准确率转成列表写入指标（行尾原注释 #.mean() 保留，表示曾考虑取均值）；用列表保留逐样本值供 log 处理。
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().tolist() #.mean()
           

        # 遍历每个负样本字段，记录正负奖励间距 margin。
        for key in rejected_rewards:
            # 记录第 key 个负样本与正样本的平均奖励差，反映偏好间隔大小。
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        # 遍历每个负样本字段，记录策略 logp 均值。
        for key in policy_rejected_logps:
            # 记录第 key 个负样本策略 logp 的批均值（detach 断开梯度），监控负样本打分变化。
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        # 记录正样本策略 logp 的批均值，与负样本对照观察偏好分离。
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()



        # 定义内嵌递归函数：把任意嵌套结构里的 torch.Tensor 转成 CPU 上的 Python 列表，便于写入指标。
        def convert_tensors_to_list(obj):
            # 若当前对象是张量，走张量转换分支。
            if isinstance(obj, torch.Tensor):
                # 脱离计算图、移到 CPU 并转成 Python 列表，避免指标持有 CUDA 张量或梯度。
                return obj.detach().cpu().tolist()
            # 若是列表，递归处理每个元素。
            elif isinstance(obj, list):
                # 对列表逐元素递归调用，保持嵌套结构不变。
                return [convert_tensors_to_list(item) for item in obj]
            # 其余类型（float、None、str 等）无需转换。
            else:
                # 原样返回非张量非列表对象（行尾两空格为原文件自带，保持不变）。
                return obj  

 
        # 把边界关键负样本的策略 logp 列表转成纯 Python 结构后记录，供 log 里原样上报。
        metrics["boundary_critical_reward"] = convert_tensors_to_list(boundary_critical_reward)
        # 把全部负样本 logp 列表转成纯 Python 结构后记录。
        metrics["all_negative_reward"] = convert_tensors_to_list(all_negative_reward)
        # 把 margin_record（多为 None 占位）转换后按固定键记录，供 log 过滤名单识别。
        metrics[f"margin_record"] = convert_tensors_to_list(margin_record)


       
        
        

        # 若边界到可辨别间距记录是非空列表（DynamicPO 分支），则展平以便统计其分布。
        if isinstance(delta_boundary2discriminative_record, list) and len(delta_boundary2discriminative_record) > 0:
            # 初始化展平列表，收集所有样本、所有边界负样本的该间距值。
            flattened_deltas = []
            # 遍历每个样本的该间距记录。
            for sample_deltas in delta_boundary2discriminative_record:
                # 若该样本记录是列表（多个边界负样本），逐项展开。
                if isinstance(sample_deltas, list):
                    # 把该样本多个间距值批量加入展平列表。
                    flattened_deltas.extend(sample_deltas)
                # 若该样本记录是单个标量，直接追加。
                else:
                    # 把单个间距值加入展平列表。
                    flattened_deltas.append(sample_deltas)

        # 同理，若正样本到边界间距记录是非空列表，则展平供统计。
        if isinstance(delta_pos2boundary_record, list) and len(delta_pos2boundary_record) > 0:
            
            # 初始化正样本到边界间距的展平列表。
            flattened_pos2boundary = []
            
            # 遍历每个样本的正样本到边界间距记录。
            for sample_deltas in delta_pos2boundary_record:
                # 该样本有多个边界负样本时逐项展开。
                if isinstance(sample_deltas, list):
                    # 把多个间距值批量加入展平列表。
                    flattened_pos2boundary.extend(sample_deltas)
                # 该样本只有一个标量间距时直接追加。
                else:
                    # 把单个间距值加入展平列表。
                    flattened_pos2boundary.append(sample_deltas)
            




        
        # 以带前缀的键记录边界到可辨别间距的均值缓存（跨行写法便于阅读）。
        metrics[f"{prefix}/delta_boundary2discriminative_mean"] = (
            # 取该均值缓存张量，移到 CPU、转 numpy 再转列表写入指标。
            self.delta_boundary2discriminative_mean.cpu().numpy().tolist()
        # 结束该指标赋值。
        )
        # 记录边界到可辨别间距的标准差缓存。
        metrics[f"{prefix}/delta_boundary2discriminative_std"] = (
            # 把标准差缓存张量转成 Python 列表写入指标。
            self.delta_boundary2discriminative_std.cpu().numpy().tolist()
        # 结束标准差指标赋值。
        )
        # 记录正样本到边界间距的均值缓存，转列表写入指标。
        metrics[f"{prefix}/delta_pos2boundary_mean"] = self.delta_pos2boundary_mean.cpu().numpy().tolist()
        # 记录正样本到边界间距的标准差缓存，转列表写入指标。
        metrics[f"{prefix}/delta_pos2boundary_std"] = self.delta_pos2boundary_std.cpu().numpy().tolist()
        # 若 beta_used 是标量 float（DMPO 分支），直接作为单值记录。
        if isinstance(beta_used, float):
            # 把标量 beta 原样赋给统一变量，后续按 float 分支写日志。
            beta_used_list_or_float = beta_used
            
        # 若 beta_used 是列表（DynamicPO 分支的逐样本动态 beta），同样直接保留。
        elif isinstance(beta_used, list):
            # 把列表原样赋给统一变量，后续按 list 分支写日志。
            beta_used_list_or_float = beta_used
        # 其余情形（张量，或 DMPO 分支占位元组里含张量）走 CPU 转换分支。
        else:
            # 把张量形式的 beta 移到 CPU、转 numpy 再转列表。
            beta_used_list_or_float = beta_used.cpu().numpy().tolist()
            # 把张量形式的正样本到边界间距记录转成 Python 列表。
            delta_pos2boundary_record = delta_pos2boundary_record.cpu().numpy().tolist()
            # 把张量形式的边界到可辨别间距记录转成 Python 列表。
            delta_boundary2discriminative_record = delta_boundary2discriminative_record.cpu().numpy().tolist()
        # 若统一后的 beta 是列表（DynamicPO 动态 beta），按列表形式写三条指标。
        if isinstance(beta_used_list_or_float, list):
            # 记录逐样本逐负样本的动态 beta 列表。
            metrics[f"{prefix}/beta_used"] = beta_used_list_or_float
            # 记录对应的正样本到边界间距记录。
            metrics[f"{prefix}/delta_pos2boundary_record"] = delta_pos2boundary_record
            # 记录对应的边界到可辨别间距记录。
            metrics[f"{prefix}/delta_boundary2discriminative_record"] = delta_boundary2discriminative_record
        # 若统一后的 beta 是标量 float（DMPO 分支），按单值形式写指标。
        elif isinstance(beta_used_list_or_float, float):
            # 把标量 beta 包成单元素列表记录，保持与列表分支结构一致。
            metrics[f"{prefix}/beta_used"] = [beta_used_list_or_float]
            # 记录 DMPO 分支的 delta_pos2boundary 占位值（beta,beta,beta 中的一项）。
            metrics[f"{prefix}/delta_pos2boundary_record"] = delta_pos2boundary_record
            # 记录 DMPO 分支的 delta_boundary2discriminative 占位值。
            metrics[f"{prefix}/delta_boundary2discriminative_record"] = delta_boundary2discriminative_record

        
        # 返回批损失均值（供 Trainer 反传）与整理好的指标字典（供 store_metrics 累积、log 上报）。
        return losses.mean(), metrics
    

    # 重写 Trainer.compute_loss：把一个批交给 _get_batch_metrics 得到 DPO 损失，并按主进程记录指标。
    def compute_loss(
        # 实例引用，用于取 accelerator、use_dpo_data_collator 与 store_metrics。
        self,
        # 当前策略模型，由 Trainer 训练循环传入。
        model: Union[PreTrainedModel, nn.Module],
        # 经 collator 整理的一个批，含 prompt、chosen 与多个 rejected 字段。
        inputs: Dict[str, Union[torch.Tensor, Any]],
        # 为 True 时额外返回指标字典，默认 False 只返回损失标量。
        return_outputs=False,
    # 返回损失张量，或 损失与指标字典 的二元组。
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # 若当前用的不是默认 DPO collator，数据格式可能不兼容，先发告警。
        if not self.use_dpo_data_collator:
            # 发出兼容性告警，提示自定义 collator 下 compute_loss 表现可能异常。
            warnings.warn(
                # 告警信息前半：compute_loss 仅针对 DPODataCollatorWithPadding 实现，传入了不同的 collator。
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                # 告警信息后半：可能出现非预期行为，若用自定义 collator 建议自行实现 prediction_step。
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            # 结束 warnings.warn 调用。
            )
        # 以 train 模式计算该批的 DMPO/DynamicPO 损失与指标，得到损失均值与指标字典。
        loss, metrics = self._get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        # 仅在主进程记录指标，避免分布式多卡重复写入同一份日志。
        if self.accelerator.is_main_process:
            # 把训练指标追加到 _stored_metrics 累积，供后续 log 统一求平均上报。
            self.store_metrics(metrics, train_eval="train")

        # 若调用方要求返回输出，则附带指标字典。
        if return_outputs:
            # 返回 损失与指标 的二元组。
            return (loss, metrics)
        # 默认只返回损失标量供 Trainer 反传。
        return loss

    # 定义 get_batch_samples：让策略模型与参考模型各对 prompt 采样生成文本，用于人工检查生成质量。
    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        # 函数文档字符串：说明本函数为给定批从策略模型与参考模型生成样本。
        """Generate samples from the model and reference model for the given batch of inputs."""

        # 调用策略模型对 prompt 做自回归采样生成推荐文本。
        policy_output = model.generate(
            # 传入 prompt 的 input_ids 作为生成起点。
            batch["prompt_input_ids"],
            # 传入 prompt 注意力掩码，标识真实 prompt token。
            attention_mask=batch["prompt_attention_mask"],
            # 生成的最大长度取配置的 max_length，防止无限生成。
            max_length=self.config.max_length,
            # 开启采样（非贪心），使生成结果带随机性、更接近真实推荐分布。
            do_sample=True,
            # 指定 pad token id，保证生成序列补齐时用同一填充符。
            pad_token_id=self.tokenizer.pad_token_id,
        # 结束策略模型 generate 调用。
        )

        # 若有参考模型，同样对 prompt 采样生成，用于与策略模型输出对比。
        reference_output = self.ref_model.generate(
            # 参考模型生成使用相同的 prompt input_ids。
            batch["prompt_input_ids"],
            # 使用相同的 prompt 注意力掩码，保证输入一致可比。
            attention_mask=batch["prompt_attention_mask"],
            # 生成最大长度同样取 self.config.max_length。
            max_length=self.config.max_length,
            # 参考模型也开启采样。
            do_sample=True,
            # 使用相同的 pad token id 补齐。
            pad_token_id=self.tokenizer.pad_token_id,
        # 仅当参考模型存在时才生成，否则整体置 None（无参考模型分支）。
        ) if self.ref_model is not None else None

        # 把策略生成结果补齐到 max_length，统一长度便于后续解码。
        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        # 批量解码为文本并跳过特殊 token，得到可读的推荐结果字符串列表。
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        # 参考输出存在时同样补齐到 max_length，否则保持 None。
        reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id) if self.ref_model is not None else None
        # 参考输出存在时批量解码为文本，否则保持 None。
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True) if self.ref_model is not None else None

        # 返回策略与参考模型解码后的生成文本二元组，供调用方对比查看。
        return policy_output_decoded, reference_output_decoded

    # 重写 Trainer.prediction_step：在评估阶段用 _get_batch_metrics 得到损失与指标，并按需返回 logits 与 labels。
    def prediction_step(
        # 实例引用。
        self,
        # 评估时传入的策略模型。
        model: Union[PreTrainedModel, nn.Module],
        # 评估批数据。
        inputs: Dict[str, Union[torch.Tensor, Any]],
        # 为 True 时只返回损失，不返回 logits 与 labels。
        prediction_loss_only: bool,
        # 推理时需忽略的输出键列表，默认 None 时从模型配置读取。
        ignore_keys: Optional[List[str]] = None,
    # 结束参数列表。
    ):
        # 非默认 DPO collator 时评估数据格式可能不兼容，先告警。
        if not self.use_dpo_data_collator:
            # 发出 prediction_step 的兼容性告警。
            warnings.warn(
                # 告警信息前半：prediction_step 仅针对 DPODataCollatorWithPadding 实现。
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                # 告警信息后半：自定义 collator 可能出现非预期行为，建议自行实现。
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            # 结束 warnings.warn 调用。
            )
        # 若调用方未指定 ignore_keys，尝试从模型配置自动获取。
        if ignore_keys is None:
            # 模型带 config 属性时才能读取推理忽略键。
            if hasattr(model, "config"):
                # 读取 config.keys_to_ignore_at_inference，取不到则默认空列表。
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            # 模型没有 config 时无法自动获取。
            else:
                # 把 ignore_keys 置为空列表，表示不忽略任何输出键。
                ignore_keys = []

        # 评估阶段关闭梯度，节省显存并避免构建计算图。
        with torch.no_grad():
            # 以 eval 模式计算该批损失与指标，指标键前缀为 eval_。
            loss, metrics = self._get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        # 仅主进程记录评估指标，避免多卡重复。
        if self.accelerator.is_main_process:
            # 把评估指标追加到 _stored_metrics 的 eval 分组。
            self.store_metrics(metrics, train_eval="eval")

        # 若只需损失，走精简返回分支。
        if prediction_loss_only:
            # 返回 脱离梯度的损失、None、None 三元组，符合 Trainer 约定。
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        # 构造 logits 字典，把评估用的 chosen logits 收进统一结构以便按 ignore_keys 过滤。
        logits_dict = {
            # 从指标字典取 logits_test/chosen 作为待返回的评估 logits。
            "logits_test/chosen": metrics["logits_test/chosen"],
            # "logits_test/rejected": metrics["logits_test/rejected"],
        # 结束 logits_dict 构造（当前仅含 chosen 一项，rejected 项在原文件中被注释掉）。
        }
        # 按 ignore_keys 过滤后把剩余 logits 值组成元组。
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        # 把元组里的 logits 堆叠后沿第 1 维求平均，得到评估用的聚合 logits。
        logits = torch.stack(logits).mean(axis=1)
        # 构造与 logits 批大小一致的全零标签占位，满足 Trainer.prediction_step 返回签名。
        labels = torch.zeros(logits.shape[0])

        # 返回 脱离梯度的损失、聚合 logits、占位标签 三元组。
        return (loss.detach(), logits, labels)

    # 定义 store_metrics：把单次前向得到的指标按 train 或 eval 分组、按指标名累积成列表，供 log 求平均。
    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        # 遍历本次指标字典的每个键值对。
        for key, value in metrics.items():
            # 把该指标值追加到对应分组、对应键的列表末尾，累积跨多个 step。
            self._stored_metrics[train_eval][key].append(value)



    # 重写 Trainer.log：在父类上报前，把 _stored_metrics 累积的指标合并进 logs，并清理已上报缓存。
    def log(self, logs: Dict[str, float]) -> None:
        # 方法文档字符串：说明 log 会把 logs（含已存储的指标）上报给所有监听训练的对象。
        # Args 之 logs：待记录的指标字典，键为指标名、值为对应数值。
        # 文档字符串结尾：以上为 log 方法的用途与参数说明。
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        # 依据 logs 里含 loss（训练）还是 eval_loss（评估）判断本次该上报哪组已存指标。
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        # 遍历该分组下每个指标名及其累积值列表。
        for key, metrics in self._stored_metrics[train_eval].items():
            # 判断该指标是否在原样上报名单（列表型或需保留结构的指标）中，名单首行给出前四个键。
            if key in ['filtered_losses', 'margin_record','losses_list','boundary_critical_reward',
                       # 名单续行：补充其余需原样上报的键，涵盖动态 beta、两段 delta 记录与多负样本准确率等；命中则不求平均。
                       'actual_losses_list','all_negative_reward','local_mask','eval_/beta_used','train_/beta_used','train_/delta_boundary2discriminative_record','train_/delta_pos2boundary_record','train_rewards/accuracies']:
                # 名单内指标直接把累积列表原样写入 logs，保留逐样本或逐负样本结构（行尾空格为原文件自带）。
                logs[key] = metrics  
            # 名单外的普通标量指标走求平均分支。
            else:
                # 把累积值转 float 张量求均值再取标量，得到该指标在本次上报区间内的平均值。
                logs[key] = torch.tensor(metrics).float().mean().item()



        # 删除已上报分组的缓存，避免下一轮 log 重复累积旧指标。
        del self._stored_metrics[train_eval]
        # 调用父类 Trainer.log 把合并后的 logs 真正写入各监控对象（如 wandb、进度条）。
        return super().log(logs)
