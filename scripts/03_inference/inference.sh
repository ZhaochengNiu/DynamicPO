#!/bin/bash
# 推理与评估阶段启动脚本，位于 SFT → 偏好优化 → 推理 流程末端。
# 加载训练好的 checkpoint，单卡调用 inference.py 生成推荐结果并由 evaluate_batch.py 计算 HitRatio@1 等指标。

# set -e 让任何命令失败立即退出，避免 checkpoint 路径错误或端口占用时仍打印误导性的评估日志。
set -e

# 用 BASH_SOURCE 取脚本目录绝对路径，使路径解析不依赖调用位置。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 用 git -C 查询仓库顶层根目录，作为 data、prompt、inference.py 等相对路径基准。
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
# 切换到仓库根目录，保证 CHECKPOINT_DIR 之类相对路径与入口脚本可正确定位。
cd "${REPO_ROOT}"

# CHECKPOINT_DIR 指向待评估的偏好优化 checkpoint（示例为 DynamicPO_DMPO/checkpoint-1122），推理时加载其 LoRA adapter。
CHECKPOINT_DIR="your_target_dir_such_as_./DynamicPO_DMPO/checkpoint-1122"
# CUDA_DEVICE=0 指定只用第 0 张 GPU，推理是单进程前向生成，无需多卡 DDP。
CUDA_DEVICE=0
# BASE_MODEL 为基础 LLM 路径，需与训练时一致，inference.py 据此加载 4bit 量化底座再合并 adapter。
BASE_MODEL="your_base_llm_model"
# DATASET=lastfm 选择测试集 lastfm-test.json，与训练数据同域以保证评估协议一致。
DATASET="lastfm"
# PROMPT_PATH 指定音乐推荐模板，作为 --external_prompt_path 传入，供推理构造与训练同构的输入。
PROMPT_PATH="./prompt/music.txt"
# BATCH_SIZE=32 是推理每卡批量，只做前向生成无需反传，可比训练批量 4 大得多以加速全量评估。
BATCH_SIZE=32

# used_ports 数组记录已占用的随机端口，供 generate_unique_port 去重，避免多次评估并行时端口撞车。
used_ports=()

# 定义 generate_unique_port 函数，循环生成并检查不重复的 master_port，保证 torchrun 能稳定启动。
generate_unique_port() {
    # while true 无限循环，直到找到一个未用过的端口才 return 退出。
    while true; do
        # local port 在 24000 到 26999 间随机取候选端口，作用域限制在函数内。
        local port=$((24000 + RANDOM % 3000))
        # if 用 [[ ]] 正则判断候选端口是否不在 used_ports 中，只有未占用才采用。
        if [[ ! " ${used_ports[*]} " =~ " $port " ]]; then
            # used_ports+=($port) 把选中的端口登记进数组，防止后续再被选中。
            used_ports+=($port)
            # echo $port 把端口输出到标准输出，命令替换 MASTER_PORT=$(...) 据此拿到值。
            echo $port
            # return 立即结束函数，避免继续循环重新生成端口。
            return
        # fi 结束端口未占用的判断块。
        fi
    # done 结束 while true 循环。
    done
# 结束 generate_unique_port 函数体。
}

# LOG_FILE 把评估日志写到 CHECKPOINT_DIR/eval.log，与被测 checkpoint 放在一起便于归档。
LOG_FILE="${CHECKPOINT_DIR}/eval.log"
# MASTER_PORT 调用 generate_unique_port 取一个不冲突的端口，供下面 torchrun 使用。
MASTER_PORT=$(generate_unique_port)

# echo 打印本次评估使用的 CUDA 设备与 master 端口，便于核对运行配置。
echo "Using CUDA device: ${CUDA_DEVICE}, Master Port: ${MASTER_PORT}"

# 启动推理主命令：CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} 只用第 0 卡，torchrun 以 --nproc_per_node 1 起单进程、--master_port=${MASTER_PORT} 用去重端口，入口在下一行单独给出 inference.py。
# inference.py 是推理入口，加载底座模型与 checkpoint，生成推荐并调用 evaluate 计算指标。
# --dataset 传 lastfm，选择 lastfm-test.json 作为测试集。
# --external_prompt_path 传音乐推荐模板，inference.py 据此设 prompt_path 并拼装测试样本输入。
# --batch_size 传 32，推理阶段单次前向的批量，加速全量测试集评估。
# --base_model 传基础 LLM，用于加载 4bit NF4 量化底座。
# --resume_from_checkpoint 传 CHECKPOINT_DIR，用 PeftModel.from_pretrained 合并偏好优化得到的 adapter；命令末行把 stdout/stderr 重定向到 ${LOG_FILE} 留存评估日志。
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" torchrun --nproc_per_node 1 --master_port="${MASTER_PORT}" \
    inference.py \
    --dataset "${DATASET}" \
    --external_prompt_path "${PROMPT_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --base_model "${BASE_MODEL}" \
    --resume_from_checkpoint "${CHECKPOINT_DIR}" > "${LOG_FILE}" 2>&1
