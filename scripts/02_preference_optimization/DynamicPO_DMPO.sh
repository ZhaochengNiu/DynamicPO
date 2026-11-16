#!/bin/bash
# DynamicPO-DMPO 主实验启动脚本，是论文核心方法在 DMPO 目标上的实现。
# 调用 DynamicPO.py 并令 filter_mode 取 DynamicPO_DMPO，触发边界关键负样本筛选与双边界动态 beta 校准分支。

# 下面英文说明原样保留，点明本脚本是 DynamicPO-DMPO 主实验，并通过 filter_mode 取 DynamicPO_DMPO 触发 dynamicpo_trainer.py 的动态分支。
# Main DynamicPO-DMPO experiment.
# This script explicitly triggers the DynamicPO-DMPO branch implemented in
# `trainer/dynamicpo_trainer.py` via `filter_mode="DynamicPO_DMPO"`.

# set -e 让任意命令失败立即退出，避免主实验在 SFT checkpoint 缺失或路径错误时仍继续跑训练。
set -e

# 用 BASH_SOURCE 取脚本所在目录绝对路径，使路径解析不依赖当前工作目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 用 git -C 查询仓库顶层根目录，作为 data、prompt、DynamicPO.py 等相对路径的基准。
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
# 切换到仓库根目录，保证后续相对路径与入口脚本能被正确定位。
cd "${REPO_ROOT}"

# GPUS 指定可见 GPU 为 0,1,2,3 共 4 张，与 NPROC_PER_NODE=4 对应，和 DMPO 基线保持同一硬件配置以便公平对比。
GPUS="0,1,2,3"
# NPROC_PER_NODE=4 起 4 个 DDP 进程，与 4 张 GPU 匹配，主实验与基线用相同并行度。
NPROC_PER_NODE=4
# MASTER_PORT 在 25000 到 26999 间随机取值，避免同一机器上并行跑基线与主实验时 DDP 端口冲突。
MASTER_PORT=$((25000 + RANDOM % 2000))

# MODEL_NAME 为基础 LLM，需与 SFT 及 DMPO 基线完全一致，保证主实验只改动算法开关而非模型。
MODEL_NAME="your_base_llm_model"
# SFT_CHECKPOINT 指向 SFT adapter，作为策略起点与参考模型来源，与 DMPO 基线共用同一 checkpoint 才能隔离 DynamicPO 的增益。
SFT_CHECKPOINT="your_sft_checkpoint"
# DATASET=lastfm 选用 LastFM，与基线一致，便于在相同数据上比较 DynamicPO 带来的 HitRatio@1 提升。
DATASET="lastfm"
# PROMPT_PATH 指定音乐推荐模板，与基线共用，确保输入构造不引入差异。
PROMPT_PATH="./prompt/music.txt"

# BETA=1.0 是动态 beta 的基准温度，与基线相同；本脚本里它会被 _build_dynamic_beta_records 按边界模糊度缩放，而非全程固定。
BETA="1.0"
# NEG_NUM=15 与基线一致，正是为了在 15 个负样本下验证 DynamicPO 能否缓解偏好优化崩塌。
NEG_NUM="15"
# BATCH_SIZE=4 与基线一致，保持有效批量相同，排除批量因素对对比结论的干扰。
BATCH_SIZE="4"
# GRAD_ACC=8 与基线一致，有效批量同为 4 卡 x 4 x 8 = 128。
GRAD_ACC="8"
# NUM_EPOCHS=3 与基线一致，训练轮数对齐以做公平对照。
NUM_EPOCHS="3"
# LR=1e-5 与基线一致，学习率不引入差异。
LR="1e-5"
# EVAL_STEP=0.033 与基线一致，约每 3.3% 步保存一次 checkpoint，便于对齐两者的评估点。
EVAL_STEP="0.033"

# FILTER_MODE=DynamicPO_DMPO 是主实验的算法开关：命中 elif filter_mode=='DynamicPO_DMPO' 分支，先用 _select_boundary_critical_samples 以 policy_chosen_logps 为阈值挑边界关键负样本（无违背时用 KMeans 聚类兜底），再由 _build_dynamic_beta_records 用双边界 delta 算样本级 beta，最后只在筛选负样本上加权算损失。
FILTER_MODE="DynamicPO_DMPO"
# LOSS_TYPE=w_ref 与基线一致，加载冻结参考模型；DynamicPO 只改负样本选择与 beta 校准，不改变是否使用 pi_ref。
LOSS_TYPE="w_ref"
# CUSTOM_NOTE=DynamicPO_DMPO 仅用于命名与打印，使主实验产物与 DMPO 基线目录分开，不参与损失计算。
CUSTOM_NOTE="DynamicPO_DMPO"
# INFO_NOTE 置空：仅主进程打印记录用，主实验无需额外标记。
INFO_NOTE=""

# OUTPUT_DIR 归到 ./DynamicPO_DMPO/dynamicpo_dmpo_neg_15_beta_1.0，把主实验结果独立成目录并编码 neg 与 beta。
OUTPUT_DIR="./DynamicPO_DMPO/dynamicpo_dmpo_neg_${NEG_NUM}_beta_${BETA}"
# WANDB_NAME 用 dynamicpo_dmpo 前缀并把 neg、beta、bs、ga 编进 run 名，便于 wandb 中与同超参的 DMPO 基线 run 直接比对。
WANDB_NAME="dynamicpo_dmpo_neg_${NEG_NUM}_beta_${BETA}_bs_${BATCH_SIZE}_ga_${GRAD_ACC}"

# 定义 model_name_to_slug，把模型路径转成小写连字符 slug，供主实验目录与 run 名按 backbone 区分。
model_name_to_slug() {
    # basename 去路径、tr 转小写、sed 把非字母数字压成连字符并去首尾连字符，得到可拼接的模型 slug。
    basename "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
# 结束 model_name_to_slug 函数体。
}

# 定义 note_to_slug，把 CUSTOM_NOTE 之类自由文本规范成可拼接 slug。
note_to_slug() {
    # 用 echo 接 tr 与 sed 把 note 里的空格和特殊字符压成连字符并转小写。
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
# 结束 note_to_slug 函数体。
}

# 先算 CUSTOM_NOTE 的 slug：本脚本为 dynamicpo-dmpo，供条件块追加到目录与 run 名。
CUSTOM_SLUG="$(note_to_slug "${CUSTOM_NOTE}")"
# 算出基础 LLM 的 slug，后面无条件追加到目录与 run 名以区分 backbone。
MODEL_SLUG="$(model_name_to_slug "${MODEL_NAME}")"
# if 判断 CUSTOM_SLUG 非空：主实验的 CUSTOM_NOTE 为 DynamicPO_DMPO，slug 为 dynamicpo-dmpo，条件成立。
if [ -n "${CUSTOM_SLUG}" ]; then
    # 把 dynamicpo-dmpo 标签追加到 OUTPUT_DIR，使主实验产物路径带方法标识，与 DMPO 基线区分。
    OUTPUT_DIR="${OUTPUT_DIR}_${CUSTOM_SLUG}"
    # 把 dynamicpo-dmpo 标签追加到 WANDB_NAME，使 wandb 中主实验 run 名带方法标识。
    WANDB_NAME="${WANDB_NAME}_${CUSTOM_SLUG}"
# fi 结束 CUSTOM_SLUG 的条件块。
fi
# 无条件把模型 slug 追加到 OUTPUT_DIR，隔离不同 backbone 的主实验结果。
OUTPUT_DIR="${OUTPUT_DIR}_${MODEL_SLUG}"
# 无条件把模型 slug 追加到 WANDB_NAME，保证 run 名唯一可追溯。
WANDB_NAME="${WANDB_NAME}_${MODEL_SLUG}"

# mkdir -p 创建主实验输出目录，确保 train.log 重定向与 checkpoint 写入时目录存在。
mkdir -p "${OUTPUT_DIR}"

# 启动主实验偏好优化命令：CUDA_VISIBLE_DEVICES=${GPUS} 用 4 卡，torchrun 以 --nproc_per_node=${NPROC_PER_NODE}（4）起 DDP、--master_port=${MASTER_PORT} 用随机端口，入口 DynamicPO.py 与 DMPO 基线相同。
# --model_name 传基础 LLM，加载 4bit 量化权重并与 SFT adapter 组装策略模型。
# --resume_from_checkpoint 传 SFT checkpoint，与基线共用同一 adapter 作为策略与参考模型起点，隔离算法增益。
# --batch_size 传 4，每卡批量与基线一致。
# --gradient_accumulation_steps 传 8，梯度累积步数与基线一致，有效批量同为 128。
# --dataset 传 lastfm，与基线共用训练/验证 json。
# --prompt_path 传音乐推荐模板，输入构造与基线一致。
# --learning_rate 传 1e-5，学习率与基线一致。
# --eval_step 传 0.033，保存/评估步比例与基线一致。
# --beta 传 1.0 作为动态 beta 的基准值；在 DynamicPO_DMPO 分支里 _build_dynamic_beta_records 会按 pos2boundary 与 boundary2discriminative 两个 delta 对每个边界负样本做 (1+delta_beta) 缩放，而非像基线那样全程固定。
# --filter_mode 传 DynamicPO_DMPO：命中 elif filter_mode=='DynamicPO_DMPO' 分支，先筛边界关键负样本再按样本级 beta 加权算 DMPO 损失，把优化重心放到近边界负样本上以缓解崩塌。
# --custom_note 传 DynamicPO_DMPO，仅用于命名与打印。
# --info_note 传空串，仅主进程打印用。
# --loss_type 传 w_ref，与基线一样加载参考模型并使用带 pi_ref 的 logratio。
# --neg_num 传 15，与基线相同，process_data 采 15 个负样本供边界筛选。
# --num_train_epochs 传 3，训练轮数与基线一致。
# --output_dir 传拼好的主实验目录，checkpoint、final_checkpoint 与 train.log 都写入其中。
# --wandb_name 传拼好的主实验 run 名，并把 stdout/stderr 重定向到 ${OUTPUT_DIR}/train.log 留存日志。
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
