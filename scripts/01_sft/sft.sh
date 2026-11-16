#!/bin/bash
# SFT 监督微调阶段启动脚本，是 DynamicPO 全流程「SFT → 偏好优化 → 推理」的第一步。
# 用 4 卡 DDP 调用 sft.py 在 LastFM 上训练 LoRA adapter，产出供偏好优化阶段 resume_from_checkpoint 加载的 SFT checkpoint。

# set -e 让任意命令失败立即终止脚本，避免 SFT 环境准备或训练出错后仍盲目启动 torchrun 产出误导性日志。
set -e

# 用 BASH_SOURCE 定位脚本自身目录并取其绝对路径，保证无论从哪个工作目录调用都能稳定解析脚本相关路径。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 通过 git -C 在脚本目录查询仓库顶层根目录，作为后续所有相对路径（data、prompt、sft.py）的基准。
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
# 切换到仓库根目录，使 ./data/lastfm-sft-cans20/、./prompt/music.txt 与 sft.py 入口都能被正确定位。
cd "${REPO_ROOT}"

# 下面英文说明原样保留，提示在 --nproc_per_node 之后填写要启动的进程数，本脚本按 4 张 GPU 填 4。
# Position the number of processes specified after the --nproc_per_node flag
# OUTPUT_DIR 基础名 ./lastfm-sft，后续拼上模型 slug，作为 SFT checkpoint、日志与 wandb 产物的根输出目录。
OUTPUT_DIR="./lastfm-sft"
# MODEL_NAME 为基础 LLM 路径或 Hugging Face 名（占位符），sft.py 据此加载 4bit NF4 量化权重并挂载 LoRA 微调。
MODEL_NAME="your_base_llm_model"

# 定义 model_name_to_slug 函数，把模型路径规范成小写连字符形式，供输出目录与 run 名按 backbone 去重。
model_name_to_slug() {
    # 取 basename 去路径、tr 转小写、sed 把非字母数字压成连字符并去首尾连字符，生成干净可拼接的模型 slug。
    basename "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
# 结束 model_name_to_slug 函数体，与上面的函数定义配对。
}

# 调用上面的函数算出模型 slug，用于给输出目录和 wandb run 名追加 backbone 标识，防止不同基座实验互相覆盖。
MODEL_SLUG="$(model_name_to_slug "${MODEL_NAME}")"
# 把模型 slug 追加到 OUTPUT_DIR，使 SFT 产物按 backbone 分目录存放，例如 ./lastfm-sft_llama2-7b-hf。
OUTPUT_DIR="${OUTPUT_DIR}_${MODEL_SLUG}"
# 用模型 slug 构造 wandb run 名，便于在 wandb 中按 backbone 区分各次 SFT 运行。
WANDB_NAME="wandb_run_name_${MODEL_SLUG}"

# mkdir -p 递归创建输出目录，确保后面把日志重定向进 ${OUTPUT_DIR}/lastfm_sft.log 时目录已存在。
mkdir -p "${OUTPUT_DIR}"


# 启动 SFT 主命令：CUDA_VISIBLE_DEVICES=0,1,2,3 指定 4 张 GPU，torchrun 以 --nproc_per_node 4 起 4 个 DDP 进程、--master_port=25664 固定通信端口，入口为 sft.py。
# --model_name 传基础 LLM，sft.py 据此加载 NF4 4bit 量化模型并套 LoRA，SFT 只训练新增低秩适配器。
# --batch_size 4 是每卡 per_device_train_batch_size，与 4 卡、梯度累积共同决定有效批量。
# --gradient_accumulation_steps 8 表示累积 8 步再更新一次，有效批量为 4 卡 x 4 x 8 = 128，兼顾显存与收敛稳定。
# --dataset lastfm 选用 LastFM 音乐推荐数据，对应 sft.py 里 ./data/lastfm-sft-cans20/ 的训练与验证 json。
# --prompt_path ./prompt/music.txt 指定音乐推荐 prompt 模板，sft.py 用它把用户历史与候选集拼成 SFT 输入。
# --logging_dir ./ 把训练日志目录设在仓库根，供 TrainingArguments 的日志回调写文件。
# --output_dir 传入已拼好模型 slug 的目录，SFT checkpoint、LoRA adapter 与最终模型都保存在这里。
# --wandb_project dpo-rec-nf4 指定 wandb 项目名，nf4 呼应全程使用的 4bit NF4 量化配置。
# --learning_rate 1e-5 是 SFT 阶段学习率，比 sft.py 默认 2e-5 更保守，避免小数据推荐微调过拟合。
# --num_train_epochs 5 训练 5 轮，比偏好优化阶段的 3 轮更多，让模型先获得粗粒度用户兴趣建模能力。
# --eval_step 0.05 作为 save_steps/eval_steps 的比例，约每 5% 训练步保存并评估一次、留存 checkpoint。
# --wandb_name 传入含模型 slug 的 run 名，使本次 SFT 在 wandb 中可追溯。
# 命令末行把 stdout 与 stderr 一并重定向到 ${OUTPUT_DIR}/lastfm_sft.log，便于后台运行时留存完整训练日志。
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25664 sft.py \
        --model_name "${MODEL_NAME}"  \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --dataset lastfm \
        --prompt_path "./prompt/music.txt" \
        --logging_dir "./" \
        --output_dir "${OUTPUT_DIR}" \
        --wandb_project dpo-rec-nf4 \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --eval_step 0.05 \
        --wandb_name "${WANDB_NAME}" \
> "${OUTPUT_DIR}/lastfm_sft.log" 2>&1
