#!/bin/bash
# 补充泛化实验中的 DynamicPO-SDPO 启动脚本，验证 DynamicPO 在 S-DPO 目标上的增益。
# 调用 exploratory_study.py，令 filter_mode 取 DynamicPO_SDPO，在 S-DPO 损失上叠加边界负样本筛选与动态 beta。

# set -e 让命令失败立即退出，避免 SFT checkpoint 缺失或端口冲突时仍继续跑补充实验。
set -e

# 用 BASH_SOURCE 取脚本目录绝对路径，保证路径解析独立于当前工作目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 用 git -C 查询仓库顶层根目录，作为 data、prompt、exploratory_study.py 相对路径基准。
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
# 切换到仓库根目录，使后续相对路径与入口脚本正确定位。
cd "${REPO_ROOT}"

# GPUS=0,1,2,3 指定 4 张可见 GPU，与 SDPO 基线相同硬件配置，保证成对对比公平。
GPUS="0,1,2,3"
# NPROC_PER_NODE=4 起 4 个 DDP 进程，与基线并行度一致。
NPROC_PER_NODE=4
# MASTER_PORT 在 25000 到 26999 间随机取值，避免同机并行跑基线与 DynamicPO 版时端口冲突。
MASTER_PORT=$((25000 + RANDOM % 2000))
# MODEL_NAME 为基础 LLM，需与 SDPO 基线一致，确保只改算法开关而非模型。
MODEL_NAME="your_base_llm_model"
# SFT_CHECKPOINT 指向 SFT adapter，与基线共用同一 checkpoint，隔离 DynamicPO 带来的增益。
SFT_CHECKPOINT="your_sft_checkpoint"
# DATASET=lastfm 与基线一致，便于在相同数据上比较 DynamicPO 对 S-DPO 的提升。
DATASET="lastfm"
# PROMPT_PATH 指定音乐推荐模板，与基线共用，确保输入构造不引入差异。
PROMPT_PATH="./prompt/music.txt"

# BETA=1.0 是动态 beta 的基准温度，与基线相同；本脚本里它会被按边界模糊度缩放，而非全程固定。
BETA="1.0"
# NEG_NUM=15 与基线一致，在相同负样本规模下验证 DynamicPO 对 S-DPO 的边界优化能力。
NEG_NUM="15"
# BATCH_SIZE=4 与基线一致，保持有效批量相同。
BATCH_SIZE="4"
# GRAD_ACC=8 与基线一致，有效批量同为 4 卡 x 4 x 8 = 128。
GRAD_ACC="8"
# NUM_EPOCHS=3 与基线一致，训练轮数对齐以做公平对照。
NUM_EPOCHS="3"
# LR=1e-5 与基线一致，学习率不引入差异。
LR="1e-5"
# CUTOFF_LEN=512 与基线一致，最大输入长度显式传入以对齐截断行为。
CUTOFF_LEN="512"
# EVAL_STEP=0.033 与基线一致，约每 3.3% 步保存 checkpoint，便于对齐评估点。
EVAL_STEP="0.033"

# OUTPUT_DIR 归到 ./outputs/exploratory_study/dynamicpo_sdpo，与 sdpo 基线目录分开，标明这是叠加 DynamicPO 的 S-DPO 实验。
OUTPUT_DIR="./outputs/exploratory_study/dynamicpo_sdpo"
# WANDB_NAME=exploratory_dynamicpo_sdpo 标明这是 DynamicPO-SDPO 的补充实验 run。
WANDB_NAME="exploratory_dynamicpo_sdpo"

# 定义 model_name_to_slug，把模型路径规范成小写连字符 slug，供目录与 run 名按 backbone 区分。
model_name_to_slug() {
    # basename、tr、sed 组合把模型路径压成干净 slug。
    basename "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
# 结束 model_name_to_slug 函数体。
}

# 定义 note_to_slug，把自由文本备注规范成可拼接 slug。
note_to_slug() {
    # 用 echo 接 tr、sed 把 note 的空格与特殊字符压成连字符并转小写。
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
# 结束 note_to_slug 函数体。
}

# CUSTOM_NOTE 取 exploratory study，与基线相同，仅作补充实验命名标签。
CUSTOM_NOTE="exploratory study"
# 算出 CUSTOM_NOTE 的 slug，供条件块追加到目录与 run 名。
CUSTOM_SLUG="$(note_to_slug "${CUSTOM_NOTE}")"
# 算出模型 slug，后面无条件追加到目录与 run 名以隔离 backbone。
MODEL_SLUG="$(model_name_to_slug "${MODEL_NAME}")"
# if 判断 CUSTOM_SLUG 非空：exploratory study 的 slug 为 exploratory-study，条件成立。
if [ -n "${CUSTOM_SLUG}" ]; then
    # 把 exploratory-study 标签追加到 OUTPUT_DIR，标明补充实验归属。
    OUTPUT_DIR="${OUTPUT_DIR}_${CUSTOM_SLUG}"
    # 把 exploratory-study 标签追加到 WANDB_NAME，使 run 名带补充实验标识。
    WANDB_NAME="${WANDB_NAME}_${CUSTOM_SLUG}"
# fi 结束条件块。
fi
# 无条件把模型 slug 追加到 OUTPUT_DIR，隔离不同 backbone 结果。
OUTPUT_DIR="${OUTPUT_DIR}_${MODEL_SLUG}"
# 无条件把模型 slug 追加到 WANDB_NAME，保证 run 名唯一可追溯。
WANDB_NAME="${WANDB_NAME}_${MODEL_SLUG}"

# mkdir -p 创建输出目录，确保 train.log 重定向与 checkpoint 写入时目录存在。
mkdir -p "${OUTPUT_DIR}"

# 启动 DynamicPO-SDPO 命令：CUDA_VISIBLE_DEVICES=${GPUS} 用 4 卡，torchrun 以 --nproc_per_node=${NPROC_PER_NODE}（4）起 DDP、--master_port=${MASTER_PORT} 用随机端口，入口 exploratory_study.py 与 SDPO 基线相同。
# --output_dir 传拼好的 dynamicpo_sdpo 目录，checkpoint 与 train.log 写入其中。
# --model_name 传基础 LLM，与基线一致。
# --prompt_path 传音乐推荐模板，与基线一致。
# --dataset 传 lastfm，与基线共用数据。
# --resume_from_checkpoint 传 SFT checkpoint，与基线共用同一 adapter。
# --wandb_name 传拼好的 DynamicPO-SDPO run 名。
# --custom_note 传 exploratory study，仅用于命名与打印。
# --info_note 传 DynamicPO-SDPO，仅作标记，标明本 run 是叠加 DynamicPO 的 S-DPO 对照。
# --beta 传 1.0 作为动态 beta 基准；在 DynamicPO_SDPO 分支里会按边界 delta 对每个筛选负样本缩放，而非基线那样全程固定。
# --filter_mode 传 DynamicPO_SDPO：命中 if filter_mode=='DynamicPO_SDPO' 分支，先以 rejected_logratios 为 target、policy_chosen_logps 为阈值调 _select_boundary_critical_samples 挑边界关键负样本，再 _build_dynamic_beta_records 算样本级 beta，最后 _compute_sdpo_losses 在筛选负样本上算 S-DPO 损失。
# --loss_type 传 w_ref：与基线一样加载参考模型、使用带 pi_ref 的 logratios，DynamicPO 只改负样本选择与 beta。
# --neg_num 传 15，与基线相同，采 15 个负样本供边界筛选。
# --batch_size 传 4，每卡训练批量与基线一致。
# --gradient_accumulation_steps 传 8，梯度累积步数与基线一致。
# --num_train_epochs 传 3，训练轮数与基线一致。
# --learning_rate 传 1e-5，学习率与基线一致。
# --cutoff_len 传 512，最大输入长度与基线一致。
# --eval_step 传 0.033，保存/评估步比例与基线一致。
# 命令末行把 stdout/stderr 重定向到 ${OUTPUT_DIR}/train.log 留存日志。
CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port=${MASTER_PORT} exploratory_study.py \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --prompt_path "${PROMPT_PATH}" \
  --dataset "${DATASET}" \
  --resume_from_checkpoint "${SFT_CHECKPOINT}" \
  --wandb_name "${WANDB_NAME}" \
  --custom_note "${CUSTOM_NOTE}" \
  --info_note "DynamicPO-SDPO" \
  --beta ${BETA} \
  --filter_mode "DynamicPO_SDPO" \
  --loss_type "w_ref" \
  --neg_num ${NEG_NUM} \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --num_train_epochs ${NUM_EPOCHS} \
  --learning_rate ${LR} \
  --cutoff_len ${CUTOFF_LEN} \
  --eval_step ${EVAL_STEP} \
  > "${OUTPUT_DIR}/train.log" 2>&1
