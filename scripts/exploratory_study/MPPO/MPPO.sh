#!/bin/bash
# 补充泛化实验中的 MPPO 基线启动脚本，用于与 DynamicPO-MPPO 成对复现对比。
# 调用 exploratory_study.py，令 filter_mode 取 MPPO、loss_type 取 wo_ref，走无参考模型、基于 exp 概率的 MPPO baseline 分支。

# set -e 让命令失败立即退出，避免 SFT checkpoint 缺失或端口冲突时仍继续跑补充实验。
set -e

# 用 BASH_SOURCE 取脚本目录绝对路径，保证路径解析独立于当前工作目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 用 git -C 查询仓库顶层根目录，作为 data、prompt、exploratory_study.py 相对路径基准。
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
# 切换到仓库根目录，使后续相对路径与入口脚本正确定位。
cd "${REPO_ROOT}"

# GPUS=0,1,2,3 指定 4 张可见 GPU，与 NPROC_PER_NODE=4 对应，用于补充实验的 DDP 训练。
GPUS="0,1,2,3"
# NPROC_PER_NODE=4 起 4 个 DDP 进程，与 4 张 GPU 一一对应。
NPROC_PER_NODE=4
# MASTER_PORT 在 25000 到 26999 间随机取值，避免同机并行实验的 DDP 端口冲突。
MASTER_PORT=$((25000 + RANDOM % 2000))
# MODEL_NAME 为基础 LLM 路径或名，需与 SFT 阶段一致，exploratory_study.py 据此加载 4bit 量化模型。
MODEL_NAME="your_base_llm_model"
# SFT_CHECKPOINT 指向 SFT adapter，作为策略模型起点；MPPO 目标 reference-free，无需用它构造参考模型。
SFT_CHECKPOINT="your_sft_checkpoint"
# DATASET=lastfm 选用 LastFM，据此读取训练/验证 json 并采样负样本。
DATASET="lastfm"
# PROMPT_PATH 指定音乐推荐模板，用于把用户历史与候选集拼成偏好优化输入。
PROMPT_PATH="./prompt/music.txt"

# BETA=1.0 是 MPPO 损失的缩放系数，作用在 exp(pi_theta) 形式的 chosen 与负样本概率差上。
BETA="1.0"
# NEG_NUM=15 是每条样本负样本数，补充实验扫描 1 到 19 观察崩塌，MPPO 相比 DMPO 能容忍更大负样本集，此处仍取 15 做统一对比。
NEG_NUM="15"
# BATCH_SIZE=4 为每卡训练批量，与 4 卡 GRAD_ACC 共同决定有效批量。
BATCH_SIZE="4"
# GRAD_ACC=8 为梯度累积步数，有效批量 4 卡 x 4 x 8 = 128。
GRAD_ACC="8"
# NUM_EPOCHS=3 训练 3 轮，与 DMPO 主实验及 SDPO 系列保持一致以便横向对比。
NUM_EPOCHS="3"
# LR=1e-5 为学习率，取小值以在 SFT 表征上做温和调整。
LR="1e-5"
# CUTOFF_LEN=512 是最大 prompt/序列长度，截断超长输入控制显存，四个补充实验均显式传它。
CUTOFF_LEN="512"
# EVAL_STEP=0.033 作为保存/评估步比例，约每 3.3% 步留存一次 checkpoint。
EVAL_STEP="0.033"

# OUTPUT_DIR 归到 ./outputs/exploratory_study/mppo，标明这是 MPPO 基线补充实验，与 SDPO 目录并列。
OUTPUT_DIR="./outputs/exploratory_study/mppo"
# WANDB_NAME=exploratory_mppo 标明这是补充实验中的 MPPO 基线 run。
WANDB_NAME="exploratory_mppo"

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

# CUSTOM_NOTE 取 exploratory study，作为四个补充实验共用的命名标签，不参与损失计算。
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

# 启动补充实验主命令：CUDA_VISIBLE_DEVICES=${GPUS} 用 4 卡，torchrun 以 --nproc_per_node=${NPROC_PER_NODE}（4）起 DDP、--master_port=${MASTER_PORT} 用随机端口，入口为 exploratory_study.py。
# --output_dir 传拼好的 mppo 目录，checkpoint、final_checkpoint 与 train.log 写入其中。
# --model_name 传基础 LLM，用于加载 4bit 量化策略模型。
# --prompt_path 传音乐推荐模板，用于构造 chosen / rejected 文本。
# --dataset 传 lastfm，选择训练/验证 json 并按 neg_num 采样负样本。
# --resume_from_checkpoint 传 SFT checkpoint，作为策略模型起点。
# --wandb_name 传拼好的 run 名，用于在 wandb 记录补充实验。
# --custom_note 传 exploratory study，仅用于命名与打印。
# --info_note 传 MPPO baseline，仅作标记，标明本 run 是 MPPO 基线对照。
# --beta 传 1.0，MPPO 损失缩放系数；基线中固定，DynamicPO 版会按边界模糊度做样本级调整。
# --filter_mode 传 MPPO：命中 trainer/exploratory_study_trainer.py 中 if filter_mode=='MPPO' 分支，调用 _compute_mppo_losses 在全部 15 个负样本上以 exp(pi_theta) 概率差算 -logsigmoid，不做边界筛选。
# --loss_type 传 wo_ref：without reference，使 exploratory_study.py 中 ref_enable 为假，不加载参考模型；MPPO 目标本身 reference-free，README 6.3 也要求 MPPO 系列用 wo_ref。
# --neg_num 传 15，process_data 为每条样本采 15 个负样本。
# --batch_size 传 4，每卡训练批量。
# --gradient_accumulation_steps 传 8，梯度累积步数。
# --num_train_epochs 传 3，训练轮数。
# --learning_rate 传 1e-5，学习率。
# --cutoff_len 传 512，最大输入长度。
# --eval_step 传 0.033，保存/评估步比例。
# 命令末行把 stdout/stderr 重定向到 ${OUTPUT_DIR}/train.log 留存日志。
CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port=${MASTER_PORT} exploratory_study.py \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --prompt_path "${PROMPT_PATH}" \
  --dataset "${DATASET}" \
  --resume_from_checkpoint "${SFT_CHECKPOINT}" \
  --wandb_name "${WANDB_NAME}" \
  --custom_note "${CUSTOM_NOTE}" \
  --info_note "MPPO baseline" \
  --beta ${BETA} \
  --filter_mode "MPPO" \
  --loss_type "wo_ref" \
  --neg_num ${NEG_NUM} \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --num_train_epochs ${NUM_EPOCHS} \
  --learning_rate ${LR} \
  --cutoff_len ${CUTOFF_LEN} \
  --eval_step ${EVAL_STEP} \
  > "${OUTPUT_DIR}/train.log" 2>&1
