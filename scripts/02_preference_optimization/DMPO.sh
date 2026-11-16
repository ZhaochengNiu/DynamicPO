#!/bin/bash
# DMPO 基线实验启动脚本，对应主实验流程中偏好优化阶段的对照基线。
# 用 4 卡 DDP 调用 DynamicPO.py，令 filter_mode 取 DMPO 触发 trainer/dynamicpo_trainer.py 中的 DMPO baseline 损失分支。

# 下面英文说明原样保留，点明本脚本是 DMPO baseline，并通过 filter_mode 取 DMPO 显式触发 dynamicpo_trainer.py 的基线分支。
# DMPO baseline experiment.
# This script explicitly triggers the baseline DMPO branch implemented in
# `trainer/dynamicpo_trainer.py` via `filter_mode="DMPO"`.

# set -e 让任意命令失败立即退出，避免偏好优化在 SFT checkpoint 缺失或路径错误时仍继续跑训练。
set -e

# 用 BASH_SOURCE 取脚本所在目录绝对路径，使路径解析不依赖当前工作目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 用 git -C 查询仓库顶层根目录，作为 data、prompt、DynamicPO.py 等相对路径的基准。
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
# 切换到仓库根目录，保证后续相对路径与入口脚本能被正确定位。
cd "${REPO_ROOT}"

# GPUS 指定可见 GPU 为 0,1,2,3 共 4 张，交给 CUDA_VISIBLE_DEVICES，和 NPROC_PER_NODE=4 一一对应。
GPUS="0,1,2,3"
# NPROC_PER_NODE=4 表示每台起 4 个 DDP 进程，与 4 张 GPU 匹配，便于分布式偏好优化训练。
NPROC_PER_NODE=4
# MASTER_PORT 在 25000 到 26999 间随机取值，避免同一机器上并行多个实验时 DDP 通信端口冲突。
MASTER_PORT=$((25000 + RANDOM % 2000))

# MODEL_NAME 为基础 LLM 路径或名，需与 SFT 阶段一致，DynamicPO.py 据此加载 4bit 量化模型并挂 LoRA。
MODEL_NAME="your_base_llm_model"
# SFT_CHECKPOINT 指向第一步产出的 SFT adapter，作为偏好优化起点并用于构造参考模型，是必须替换的占位符。
SFT_CHECKPOINT="your_sft_checkpoint"
# DATASET=lastfm 选用 LastFM，DynamicPO.py 据此读取 lastfm-train.json / lastfm-val.json 并采样负样本。
DATASET="lastfm"
# PROMPT_PATH 指定音乐推荐 prompt 模板，用于把用户历史与候选物品拼成偏好优化输入。
PROMPT_PATH="./prompt/music.txt"

# BETA=1.0 是 DPO/DMPO 损失温度系数，控制策略对 chosen 与 rejected 概率比差异的敏感度，越大越约束偏离参考模型。
BETA="1.0"
# NEG_NUM=15 是每条样本采样的负样本数，README 主实验固定为 15；负样本过多会诱发论文研究的偏好优化崩塌。
NEG_NUM="15"
# BATCH_SIZE=4 是每卡训练批量，与 4 卡 GRAD_ACC 一起决定有效批量大小。
BATCH_SIZE="4"
# GRAD_ACC=8 为梯度累积步数，有效批量 4 卡 x 4 x 8 = 128，在有限显存下维持较大批量。
GRAD_ACC="8"
# NUM_EPOCHS=3 训练 3 轮，比 SFT 的 5 轮少，偏好优化只需在 SFT 基础上做边界微调。
NUM_EPOCHS="3"
# LR=1e-5 是偏好优化学习率，与 SFT 一致取较小值，避免破坏 SFT 已学到的用户兴趣表征。
LR="1e-5"
# EVAL_STEP=0.033 作为 save_steps/eval_steps 比例，约每 3.3% 步保存一次 checkpoint，便于挑选最佳模型做推理。
EVAL_STEP="0.033"

# FILTER_MODE=DMPO 是算法开关：传入 dynamicpo_trainer.preference_loss 后命中 if filter_mode=='DMPO' 分支，对 15 个负样本 logratio 等权平均算 -logsigmoid，不做边界筛选也不做动态 beta，是纯基线。
FILTER_MODE="DMPO"
# LOSS_TYPE=w_ref 表示 with reference：DynamicPO.py 中 ref_enable=loss_type not in [wo_ref] 为真，额外加载冻结的参考模型，DMPO 目标含 pi_ref 概率比。
LOSS_TYPE="w_ref"
# CUSTOM_NOTE=DMPO 仅用于拼接输出目录与 wandb run 名并打印，不参与损失计算；Python 端 _append_note_suffix 会因目录已含该 slug 而避免重复追加。
CUSTOM_NOTE="DMPO"
# INFO_NOTE 置空：仅在主进程打印记录用，基线实验无需额外标记。
INFO_NOTE=""

# OUTPUT_DIR 用 neg_num 与 beta 拼出 ./DMPO/dmpo_neg_15_beta_1.0，把本基线结果归到 DMPO 目录并编码关键超参。
OUTPUT_DIR="./DMPO/dmpo_neg_${NEG_NUM}_beta_${BETA}"
# WANDB_NAME 把 neg、beta、bs、ga 都编进 run 名，便于在 wandb 里按超参横向对比多次基线实验。
WANDB_NAME="dmpo_neg_${NEG_NUM}_beta_${BETA}_bs_${BATCH_SIZE}_ga_${GRAD_ACC}"

# 定义 model_name_to_slug，把模型路径转成小写连字符 slug，供输出目录/run 名按 backbone 区分。
model_name_to_slug() {
    # basename 去路径、tr 转小写、sed 把非字母数字压成连字符并去首尾连字符，得到可拼接的模型 slug。
    basename "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
# 结束 model_name_to_slug 函数体。
}

# 定义 note_to_slug，把 CUSTOM_NOTE 之类自由文本也规范成可拼接的 slug，用于目录与 run 名。
note_to_slug() {
    # 与模型 slug 类似，用 echo 接 tr 与 sed 把 note 里的空格和特殊字符压成连字符并转小写。
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
# 结束 note_to_slug 函数体。
}

# 先算 CUSTOM_NOTE 的 slug，供下面 if 判断是否给目录与 run 名追加实验标签。
CUSTOM_SLUG="$(note_to_slug "${CUSTOM_NOTE}")"
# 算出基础 LLM 的 slug，后面无条件追加到目录与 run 名以区分 backbone。
MODEL_SLUG="$(model_name_to_slug "${MODEL_NAME}")"
# if 判断 CUSTOM_SLUG 非空：DMPO 的 CUSTOM_NOTE 为 DMPO，slug 为 dmpo，条件成立会进入分支。
if [ -n "${CUSTOM_SLUG}" ]; then
    # 把 dmpo 标签追加到 OUTPUT_DIR，使基线产物路径带上方法标识，与 DynamicPO 版区分。
    OUTPUT_DIR="${OUTPUT_DIR}_${CUSTOM_SLUG}"
    # 把 dmpo 标签追加到 WANDB_NAME，使 wandb 中基线 run 名带方法标识。
    WANDB_NAME="${WANDB_NAME}_${CUSTOM_SLUG}"
# fi 结束 CUSTOM_SLUG 的条件块。
fi
# 无条件把模型 slug 追加到 OUTPUT_DIR，隔离不同 backbone 的实验结果。
OUTPUT_DIR="${OUTPUT_DIR}_${MODEL_SLUG}"
# 无条件把模型 slug 追加到 WANDB_NAME，保证 run 名全局唯一且可追溯。
WANDB_NAME="${WANDB_NAME}_${MODEL_SLUG}"

# mkdir -p 创建最终输出目录，确保 train.log 重定向与 checkpoint 写入时目录存在。
mkdir -p "${OUTPUT_DIR}"

# 启动偏好优化主命令：CUDA_VISIBLE_DEVICES=${GPUS} 用 4 张 GPU，torchrun 以 --nproc_per_node=${NPROC_PER_NODE}（4）起 DDP 进程、--master_port=${MASTER_PORT} 用随机端口，入口为 DynamicPO.py。
# --model_name 传基础 LLM，用于加载 4bit 量化权重并与 SFT adapter 组装策略模型。
# --resume_from_checkpoint 指向 SFT checkpoint，DynamicPO.py 用 PeftModel.from_pretrained 加载它作为可训练策略与冻结参考模型起点。
# --batch_size 传 4，对应每卡批量，写入 TrainingArguments.per_device_train_batch_size。
# --gradient_accumulation_steps 传 8，与批量、卡数共同给出有效批量 128。
# --dataset 传 lastfm，选择对应训练/验证 json 并据 neg_num 采样负样本。
# --prompt_path 传音乐推荐模板，用于构造 chosen / rejected 文本。
# --learning_rate 传 1e-5，偏好优化阶段的小学习率。
# --eval_step 传 0.033，作为按步保存与评估的比例。
# --beta 传 1.0，DPO 温度系数；基线中固定不变，DynamicPO 版才会按边界模糊度做样本级调整。
# --filter_mode 传 DMPO：命中 trainer/dynamicpo_trainer.py 的 if filter_mode=='DMPO' 分支，用 K 个负样本 logratio 的均值算 -logsigmoid，是全负样本等权的基线损失。
# --custom_note 传 DMPO，仅用于命名与打印，Python 端据此追加目录后缀（已含则跳过）。
# --info_note 传空串，仅在主进程打印，不影响算法。
# --loss_type 传 w_ref，令 DynamicPO.py 构造并加载参考模型，使损失使用带 pi_ref 的 logratio。
# --neg_num 传 15，process_data 为每个样本随机采 15 个负样本并生成 rejected1 到 rejected15 字段。
# --num_train_epochs 传 3，控制偏好优化训练轮数。
# --output_dir 传拼好的目录，checkpoint、final_checkpoint 与 train.log 都写入其中。
# --wandb_name 传拼好的 run 名，并把 stdout/stderr 重定向到 ${OUTPUT_DIR}/train.log 留存训练日志。
CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port=${MASTER_PORT} DynamicPO.py \
    --model_name "${MODEL_NAME}" \
    --resume_from_checkpoint "${SFT_CHECKPOINT}" \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --dataset "${DATASET}" \
    --prompt_path "${PROMPT_PATH}" \
    --learning_rate ${LR} \
    --eval_step ${EVAL_STEP} \
    --beta ${BETA} \
    --filter_mode "${FILTER_MODE}" \
    --custom_note "${CUSTOM_NOTE}" \
    --info_note "${INFO_NOTE}" \
    --loss_type "${LOSS_TYPE}" \
    --neg_num ${NEG_NUM} \
    --num_train_epochs ${NUM_EPOCHS} \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_name "${WANDB_NAME}" > "${OUTPUT_DIR}/train.log" 2>&1
