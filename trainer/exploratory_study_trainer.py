# 探索性实验用的自定义 Trainer 与损失实现，覆盖 MPPO、S-DPO 及其 DynamicPO 版本的偏好损失计算与指标记录
# 继承主实验 DPOTrainer，复用其前向与动态 beta 工具，在此按 filter_mode 分派不同目标并汇总训练与评估指标
# 导入 torch，供张量类型标注、指数运算与损失堆叠等操作使用
import torch
# 导入函数式接口并以 F 为别名，主要使用 logsigmoid 构造偏好损失
import torch.nn.functional as F
# 导入类型标注工具，用于标注张量、字典、列表与 train/eval 字面量类型
from typing import Any, Dict, List, Literal, Tuple, Union

# 从主实验 trainer 导入基类与动态负样本工具，避免重复实现
from .dynamicpo_trainer import (
    # 把主实验 DPOTrainer 作为基类，复用其前向、数据整理与训练循环
    DPOTrainer as BaseDPOTrainer,
    # 导入动态 beta 记录构建函数，按边界歧义度校准样本级优化强度
    _build_dynamic_beta_records,
    # 导入按样本分组的工具，把 rejected 列字典重排为逐样本的列表
    _group_batch_values,
    # 导入边界关键负样本筛选，优先偏好违反负样本并在无违反时做似然聚类
    _select_boundary_critical_samples,
# 结束从主实验 trainer 的导入列表
)


# 定义 DMPO 目标的损失计算，按 beta 是否为逐样本列表分两种形式
def _compute_dmpo_losses(
    # 正样本的策略与参考 log 概率比，逐样本张量
    chosen_logratios: torch.FloatTensor,
    # 每个样本筛选出的负样本 logratio 列表，供多负样本聚合
    boundary_critical_targets: List[List[torch.Tensor]],
    # beta 既可为全局浮点，也可为逐样本逐负样本的动态列表
    beta_values: Union[float, List[List[float]]],
# 函数签名收尾，返回堆叠后的逐样本损失张量
) -> torch.FloatTensor:
    # 初始化逐样本损失列表
    batch_losses = []
    # 逐样本遍历其负样本集合，i 为样本下标
    for i, sample_negatives in enumerate(boundary_critical_targets):
        # 动态 beta 分支，beta 为逐样本逐负样本列表
        if isinstance(beta_values, list):
            # 取该样本负样本个数 k，用于按均值归一化
            k = len(sample_negatives)
            # 损失取负的 log-sigmoid，符合 DMPO 的偏好排序目标
            loss_value = -F.logsigmoid(
                # 对负样本逐项加权后求和
                sum(
                    # 正样本项为动态 beta 乘正样本 logratio 再按 k 摊薄
                    beta_values[i][idx] * chosen_logratios[i] / k
                    # 负样本项为同权重乘各负样本 logratio 并按 k 摊薄后相减
                    - beta_values[i][idx] * neg_sample / k
                    # 枚举该样本每个负样本及其下标，取对应动态 beta
                    for idx, neg_sample in enumerate(sample_negatives)
                # 结束负样本加权求和
                )
            # 结束 log-sigmoid 内的表达式
            )
        # 固定标量 beta 分支
        else:
            # 取负样本个数 k 用于平均
            k = len(sample_negatives)
            # 同样取负的 log-sigmoid 作为偏好损失
            loss_value = -F.logsigmoid(
                # 标量 beta 乘当前样本的正样本 logratio
                beta_values * chosen_logratios[i]
                # 减去所有负样本经 beta 加权后的均值，实现多负样本平均
                - sum(beta_values * neg_sample for neg_sample in sample_negatives) / k
            # 结束 log-sigmoid 表达式
            )
        # 把当前样本损失加入批次列表
        batch_losses.append(loss_value)
    # 堆叠为张量返回，供上层取均值
    return torch.stack(batch_losses)


# 定义 MPPO 目标的损失计算，直接用策略概率而非 logratio
def _compute_mppo_losses(
    # 正样本的策略 log 概率，逐样本张量
    policy_chosen_logps: torch.FloatTensor,
    # 每个样本的负样本 log 概率列表
    boundary_critical_targets: List[List[torch.Tensor]],
    # beta 可为标量或动态列表，缩放各项权重
    beta_values: Union[float, List[List[float]]],
# 签名收尾，返回逐样本损失张量
) -> torch.FloatTensor:
    # 初始化逐样本损失列表
    batch_losses = []
    # 逐样本遍历负样本集合
    for i, sample_negatives in enumerate(boundary_critical_targets):
        # 动态 beta 分支
        if isinstance(beta_values, list):
            # 损失为负的 log-sigmoid，对应 MPPO 的 N 倍正样本概率减去负样本概率之和
            loss_value = -F.logsigmoid(
                # 对正负样本的概率差逐项加权求和
                sum(
                    # 正样本项为动态 beta 乘正样本策略概率，exp 还原 logps
                    beta_values[i][idx] * torch.exp(policy_chosen_logps[i])
                    # 负样本项为同权重乘各负样本策略概率后相减
                    - beta_values[i][idx] * torch.exp(neg_sample)
                    # 枚举负样本以取对应的逐负样本 beta
                    for idx, neg_sample in enumerate(sample_negatives)
                # 结束加权求和
                )
            # 结束 log-sigmoid 表达式
            )
        # 固定标量 beta 分支
        else:
            # 取负样本个数 k，用于按 MPPO 公式放大正样本项
            k = len(sample_negatives)
            # 同样取负的 log-sigmoid
            loss_value = -F.logsigmoid(
                # 正样本项为 beta 乘 k 乘正样本策略概率，对应 MPPO 中 N 倍正样本概率
                beta_values * k * torch.exp(policy_chosen_logps[i])
                # 减去所有负样本经 beta 加权的概率之和
                - sum(beta_values * torch.exp(neg_sample) for neg_sample in sample_negatives)
            # 结束 log-sigmoid 表达式
            )
        # 把当前样本损失加入批次列表
        batch_losses.append(loss_value)
    # 堆叠为张量返回
    return torch.stack(batch_losses)


# 定义 S-DPO 目标的损失计算，用 softmax 形式聚合多负样本
def _compute_sdpo_losses(
    # 正样本的策略与参考 log 概率比
    chosen_logratios: torch.FloatTensor,
    # 每个样本的负样本 logratio 列表
    boundary_critical_targets: List[List[torch.Tensor]],
    # beta 可为标量或动态列表
    beta_values: Union[float, List[List[float]]],
# 签名收尾，返回逐样本损失张量
) -> torch.FloatTensor:
    # 初始化逐样本损失列表
    batch_losses = []
    # 逐样本遍历负样本集合
    for i, sample_negatives in enumerate(boundary_critical_targets):
        # 动态 beta 分支
        if isinstance(beta_values, list):
            # 累加各负样本相对正样本的指数项，构造 softmax 的分母和
            temp = sum(
                # 逐负样本用动态 beta 缩放其与正样本 logratio 之差再取指数
                torch.exp(beta_values[i][idx] * (neg_sample - chosen_logratios[i]))
                # 枚举负样本以取对应动态 beta
                for idx, neg_sample in enumerate(sample_negatives)
            # 结束指数项求和
            )
        # 固定标量 beta 分支
        else:
            # 标量 beta 下累加所有 exp(beta 乘负样本与正样本之差) 项
            temp = sum(torch.exp(beta_values * (neg_sample - chosen_logratios[i])) for neg_sample in sample_negatives)
        # 取负的 logsigmoid 作用于负的 log(temp 加 epsilon)，加 1e-8 防止 temp 为 0 时对数溢出
        loss_value = -F.logsigmoid(-torch.log(temp + 1e-8))
        # 把当前样本损失加入批次列表
        batch_losses.append(loss_value)
    # 堆叠为张量返回
    return torch.stack(batch_losses)


# 定义偏好损失分派入口，按 filter_mode 选择 MPPO、S-DPO 及其 DynamicPO 版本
def preference_loss(
    # 是否启用参考模型，决定 logratio 是否扣除参考概率
    ref_model_enabled: bool,
    # 策略模型对正样本的 log 概率
    policy_chosen_logps: torch.FloatTensor,
    # 策略模型对多个 rejected 列的 log 概率字典
    policy_rejected_logps: Dict[str, torch.FloatTensor],
    # 参考模型对正样本的 log 概率，wo_ref 时为 None
    reference_chosen_logps: torch.FloatTensor,
    # 参考模型对多个 rejected 列的 log 概率字典
    reference_rejected_logps: Dict[str, torch.FloatTensor],
    # 目标模式，决定走哪条损失与动态筛选分支
    filter_mode: str,
    # 全局温度系数 beta，作为动态 beta 调整的基准
    beta: float,
# 结束参数列表，函数体开始
):
    # 正样本 logratio，有参考模型时取策略减参考，否则直接用策略 logps
    chosen_logratios = policy_chosen_logps - reference_chosen_logps if ref_model_enabled else policy_chosen_logps
    # 构造正样本奖励，仅用于指标记录，需 detach 阻断梯度
    chosen_rewards = (
        # 带参考模型时奖励为 beta 乘策略与参考之差并 detach
        beta * (policy_chosen_logps - reference_chosen_logps).detach()
        # 以是否启用参考模型作为三元表达式条件
        if ref_model_enabled
        # 无参考模型时奖励为 beta 乘策略 logps 并 detach
        else beta * policy_chosen_logps.detach()
    # 结束正样本奖励的三元构造
    )

    # 初始化各 rejected 列的 logratio 字典
    rejected_logratios = {}
    # 逐 rejected 列计算其策略与参考差值
    for key in policy_rejected_logps:
        # 构造当前列的 logratio
        rejected_logratios[key] = (
            # 带参考模型时取策略减参考的 log 概率
            policy_rejected_logps[key] - reference_rejected_logps[key]
            # 三元条件为是否启用参考模型
            if ref_model_enabled
            # 无参考模型时直接用策略 rejected logps
            else policy_rejected_logps[key]
        # 结束当前 rejected 列的 logratio 构造
        )

    # 初始化各 rejected 列的奖励字典，供指标记录
    rejected_rewards = {}
    # 逐 rejected 列计算奖励
    for key in policy_rejected_logps:
        # 构造当前列的奖励
        rejected_rewards[key] = (
            # 带参考模型时以 beta 乘差值并 detach
            beta * (policy_rejected_logps[key] - reference_rejected_logps[key]).detach()
            # 三元条件为是否启用参考模型
            if ref_model_enabled
            # 无参考模型时以 beta 乘策略 logps 并 detach
            else beta * policy_rejected_logps[key].detach()
        # 结束当前 rejected 奖励构造
        )

    # 把 rejected 列字典按样本分组，得到逐样本的负样本 logratio 列表
    batch_rejected_logratios = _group_batch_values(rejected_logratios)
    # 转成列表形式，供 S-DPO 系损失按样本访问
    all_rejected_logratios = [batch_rejected_logratios[i] for i in range(len(batch_rejected_logratios))]
    # 把策略 rejected logps 同样按样本分组
    all_policy_rejected = _group_batch_values(policy_rejected_logps)
    # 转成列表，供 MPPO 系损失按样本访问
    all_policy_rejected_list = [all_policy_rejected[i] for i in range(len(all_policy_rejected))]

    # MPPO 基线分支，使用全部负样本与固定标量 beta
    if filter_mode == "MPPO":
        # 按 MPPO 公式计算逐样本损失
        losses = _compute_mppo_losses(policy_chosen_logps, all_policy_rejected_list, beta)
        # 返回损失与空动态记录，beta 记录用三份标量占位
        return (losses, None), None, None, chosen_rewards, rejected_rewards, (beta, beta, beta)

    # DynamicPO-MPPO 分支，先筛边界关键负样本再动态调 beta
    if filter_mode == "DynamicPO_MPPO":
        # 接收边界负样本、其策略值、模型可分负样本均值与整批 rejected 列表
        boundary_critical_targets, boundary_critical_policy_values, model_discriminative_policy_means, batch_policy_rejected_list = (
            # 调用边界关键负样本筛选，优先偏好违反样本并做似然聚类
            _select_boundary_critical_samples(
                # 以策略 rejected logps 作为待筛选的目标值
                target_values=policy_rejected_logps,
                # 以正样本 logratio 作为判断偏好是否被违反的阈值
                threshold_values=chosen_logratios,
                # 传入策略 rejected 值用于计算模型可分性统计
                policy_rejected_values=policy_rejected_logps,
            # 结束筛选函数调用
            )
        # 结束四元组解包
        )
        # 按边界歧义度构建逐样本动态 beta 及两类间隔记录
        beta_records, pos2boundary_records, boundary2discriminative_records = _build_dynamic_beta_records(
            # 传入正样本策略 logps 作为双间隔的基准
            policy_chosen_logps,
            # 传入筛选出的边界负样本策略值
            boundary_critical_policy_values,
            # 传入模型可分负样本均值，用于校准优化强度
            model_discriminative_policy_means,
            # 传入全局 beta 作为动态缩放基准
            beta,
        # 结束动态 beta 记录构建
        )
        # 仅用边界关键负样本与动态 beta 计算 MPPO 损失
        losses = _compute_mppo_losses(policy_chosen_logps, boundary_critical_targets, beta_records)
        # 返回 DynamicPO-MPPO 的损失与全部动态记录
        return (
            # 损失与空 margin 记录，探索性目标未使用 margin
            (losses, None),
            # 返回边界负样本策略值供指标记录
            boundary_critical_policy_values,
            # 返回整批 rejected 值供全负样本奖励统计
            batch_policy_rejected_list,
            # 返回正样本奖励
            chosen_rewards,
            # 返回各 rejected 奖励
            rejected_rewards,
            # 返回动态 beta 与两类间隔记录
            (beta_records, pos2boundary_records, boundary2discriminative_records),
        # 结束返回元组
        )

    # S-DPO 基线分支，兼容 SDPO 与 S-DPO 两种写法
    if filter_mode in {"SDPO", "S-DPO"}:
        # 按 softmax 多负样本形式计算损失
        losses = _compute_sdpo_losses(chosen_logratios, all_rejected_logratios, beta)
        # 返回损失与空动态记录，beta 用三份标量占位
        return (losses, None), None, None, chosen_rewards, rejected_rewards, (beta, beta, beta)

    # DynamicPO-SDPO 分支，筛边界关键负样本并动态调 beta
    if filter_mode == "DynamicPO_SDPO":
        # 接收边界负样本、其策略值、可分负样本均值与整批 rejected 列表
        boundary_critical_targets, boundary_critical_policy_values, model_discriminative_policy_means, batch_policy_rejected_list = (
            # 调用边界关键负样本筛选
            _select_boundary_critical_samples(
                # S-DPO 以策略与参考的 rejected logratio 作为筛选目标
                target_values=rejected_logratios,
                # 以策略正样本 logps 作为偏好违反判断阈值
                threshold_values=policy_chosen_logps,
                # 传入策略 rejected 值用于可分性统计
                policy_rejected_values=policy_rejected_logps,
            # 结束筛选函数调用
            )
        # 结束四元组解包
        )
        # 构建逐样本动态 beta 与两类间隔记录
        beta_records, pos2boundary_records, boundary2discriminative_records = _build_dynamic_beta_records(
            # 传入正样本策略 logps 作为基准
            policy_chosen_logps,
            # 传入边界负样本策略值
            boundary_critical_policy_values,
            # 传入模型可分负样本均值
            model_discriminative_policy_means,
            # 传入全局 beta 作为缩放基准
            beta,
        # 结束动态 beta 记录构建
        )
        # 仅用边界关键负样本与动态 beta 计算 S-DPO 损失
        losses = _compute_sdpo_losses(chosen_logratios, boundary_critical_targets, beta_records)
        # 返回 DynamicPO-SDPO 的损失与全部动态记录
        return (
            # 损失与空 margin 记录
            (losses, None),
            # 返回边界负样本策略值
            boundary_critical_policy_values,
            # 返回整批 rejected 值
            batch_policy_rejected_list,
            # 返回正样本奖励
            chosen_rewards,
            # 返回各 rejected 奖励
            rejected_rewards,
            # 返回动态 beta 与两类间隔记录
            (beta_records, pos2boundary_records, boundary2discriminative_records),
        # 结束返回元组
        )

    # 未匹配到支持的 filter_mode 时抛错，避免静默走空分支
    raise ValueError(f"Unsupported filter_mode for exploratory study trainer: {filter_mode}")


# 定义探索性实验 Trainer，继承主实验 DPOTrainer 以复用训练循环
class DPOTrainer(BaseDPOTrainer):
    # 构造函数额外接收 loss_type，其余参数透传给基类
    def __init__(self, *args, loss_type: str = "", **kwargs):
        # 保存 w_ref 与 wo_ref 标记，供后续判断是否计算参考 logps
        self.loss_type = loss_type
        # 调用基类初始化，建立模型、数据整理器与动态 beta 统计属性
        super().__init__(*args, **kwargs)

    # 重写批次指标计算，按当前 filter_mode 分派损失并汇总奖励与 metrics
    def _get_batch_metrics(
        # 实例自身
        self,
        # 当前用于前向的模型，训练时为策略模型
        model,
        # 经 collator 整理的批次，含 prompt、chosen、rejected 各列张量
        batch: Dict[str, Union[List, torch.LongTensor]],
        # 标记处于 train 还是 eval，用于给指标加前缀
        train_eval: Literal["train", "eval"] = "train",
    # 结束函数签名
    ):
        # 初始化指标字典
        metrics = {}

        # 开始解包策略模型前向返回的四元组
        (
            # 策略模型对正样本的 log 概率
            policy_chosen_logps,
            # 策略模型对各 rejected 列 log 概率的字典
            policy_rejected_logps,
            # 策略模型正样本 logits，此处不用于损失
            policy_chosen_logits,
            # 策略模型各 rejected logits，此处不用于损失
            policy_rejected_logits,
        # 调用基类前向，一次性拼接 chosen 与 rejected 计算策略 logps 与 logits
        ) = self.concatenated_forward(model, batch)

        # 参考模型前向不参与梯度，关闭 autograd 以省显存
        with torch.no_grad():
            # 开始解包参考模型前向返回的四元组
            (
                # 参考模型对正样本的 log 概率
                reference_chosen_logps,
                # 参考模型对各 rejected 列的 log 概率字典
                reference_rejected_logps,
                # 参考模型正样本 logits，占位忽略
                _,
                # 参考模型 rejected logits，占位忽略
                _,
            # 有参考模型则计算其 logps，无参考模型的 MPPO 系用四个 None 占位
            ) = self.concatenated_forward(self.ref_model, batch) if self.ref_model is not None else (None, None, None, None)

        # 解包 preference_loss 返回的损失、奖励与动态记录
        (losses, margin_record), boundary_critical_reward, all_negative_reward, chosen_rewards, rejected_rewards, (
            # 本次实际使用的 beta，标量或动态列表
            beta_used,
            # 正样本到边界负样本的间隔记录
            delta_pos2boundary_record,
            # 边界负样本到模型可分负样本的间隔记录
            delta_boundary2discriminative_record,
        # 调用偏好损失分派，按 filter_mode 计算损失与各项记录
        ) = preference_loss(
            # 以是否存在参考模型决定是否扣除参考概率
            ref_model_enabled=self.ref_model is not None,
            # 传入策略正样本 logps
            policy_chosen_logps=policy_chosen_logps,
            # 传入策略各 rejected logps
            policy_rejected_logps=policy_rejected_logps,
            # 传入参考正样本 logps，wo_ref 时为 None
            reference_chosen_logps=reference_chosen_logps,
            # 传入参考各 rejected logps，wo_ref 时为 None
            reference_rejected_logps=reference_rejected_logps,
            # 传入当前目标模式决定损失分支
            filter_mode=self.filter_mode,
            # 传入全局 beta 作为动态缩放基准
            beta=self.beta,
        # 结束 preference_loss 调用
        )

        # 初始化奖励准确率，衡量 chosen 是否大于所有 rejected
        reward_accuracies = None
        # 逐 rejected 列累乘判断
        for key in rejected_rewards:
            # 首个 rejected 列时初始化准确率张量
            if reward_accuracies is None:
                # 比较正样本奖励是否大于该 rejected 奖励，转为 0 或 1
                reward_accuracies = (chosen_rewards > rejected_rewards[key]).float()
            # 后续 rejected 列以累乘方式合并
            else:
                # 要求正样本奖励大于全部负样本才算命中
                reward_accuracies *= (chosen_rewards > rejected_rewards[key]).float()

        # 按 train 或 eval 选择指标前缀，区分训练与评估日志
        prefix = "eval_" if train_eval == "eval" else "train_"
        # 记录正样本奖励均值
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        # 逐 rejected 列记录奖励均值
        for key in rejected_rewards:
            # 把该列奖励均值写入指标
            metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()

        # 记录正样本胜过全部负样本的比例
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().tolist()
        # 逐 rejected 列记录奖励间隔
        for key in rejected_rewards:
            # 记录正样本与该列负样本的奖励差均值
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        # 逐 rejected 列记录策略 logps
        for key in policy_rejected_logps:
            # 记录该列策略 log 概率均值并 detach
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        # 记录正样本策略 log 概率均值并 detach
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()

        # 定义递归工具，把嵌套结构中的张量转成普通 Python 列表以便写入日志
        def convert_tensors_to_list(obj: Any):
            # 张量分支，detach 后搬到 CPU 并转列表
            if isinstance(obj, torch.Tensor):
                # 返回脱离计算图的 CPU 列表
                return obj.detach().cpu().tolist()
            # 列表分支，逐元素递归处理
            if isinstance(obj, list):
                # 递归转换列表内每个元素
                return [convert_tensors_to_list(item) for item in obj]
            # 其余类型含 None 与标量，原样返回
            return obj

        # 记录边界关键负样本相关奖励并转成可序列化列表
        metrics["boundary_critical_reward"] = convert_tensors_to_list(boundary_critical_reward)
        # 记录全负样本奖励统计
        metrics["all_negative_reward"] = convert_tensors_to_list(all_negative_reward)
        # 记录 margin 记录，探索性目标通常为 None
        metrics["margin_record"] = convert_tensors_to_list(margin_record)
        # 记录边界到可分负样本间隔的均值
        metrics[f"{prefix}/delta_boundary2discriminative_mean"] = (
            # 取基类统计的对应均值并转列表
            self.delta_boundary2discriminative_mean.cpu().numpy().tolist()
        # 结束该指标赋值
        )
        # 记录边界到可分负样本间隔的标准差
        metrics[f"{prefix}/delta_boundary2discriminative_std"] = (
            # 取基类统计的对应标准差并转列表
            self.delta_boundary2discriminative_std.cpu().numpy().tolist()
        # 结束该指标赋值
        )
        # 记录正样本到边界负样本间隔的均值
        metrics[f"{prefix}/delta_pos2boundary_mean"] = self.delta_pos2boundary_mean.cpu().numpy().tolist()
        # 记录正样本到边界负样本间隔的标准差
        metrics[f"{prefix}/delta_pos2boundary_std"] = self.delta_pos2boundary_std.cpu().numpy().tolist()

        # beta 为标量浮点时直接保留
        if isinstance(beta_used, float):
            # 标量 beta 原样赋给中间变量
            beta_used_list_or_float = beta_used
        # beta 为动态列表时同样保留
        elif isinstance(beta_used, list):
            # 列表 beta 原样赋给中间变量
            beta_used_list_or_float = beta_used
        # 其余情况如张量需转成列表
        else:
            # 把张量形式的 beta 搬到 CPU 转列表
            beta_used_list_or_float = beta_used.cpu().numpy().tolist()

        # 动态 beta 以列表形式记录
        if isinstance(beta_used_list_or_float, list):
            # 写入逐样本动态 beta 列表
            metrics[f"{prefix}/beta_used"] = beta_used_list_or_float
        # 标量 beta 包成单元素列表
        elif isinstance(beta_used_list_or_float, float):
            # 写入含单一 beta 的列表，保持指标形状一致
            metrics[f"{prefix}/beta_used"] = [beta_used_list_or_float]

        # 记录正样本到边界负样本间隔的逐样本明细
        metrics[f"{prefix}/delta_pos2boundary_record"] = delta_pos2boundary_record
        # 记录边界到可分负样本间隔的逐样本明细
        metrics[f"{prefix}/delta_boundary2discriminative_record"] = delta_boundary2discriminative_record

        # 返回批次平均损失供反向传播，以及指标字典供日志
        return losses.mean(), metrics
