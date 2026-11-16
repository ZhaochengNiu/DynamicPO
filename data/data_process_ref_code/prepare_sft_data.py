# 文件级说明：prepare_sft_data.py 是数据集预处理的导出脚本（参考代码），把三个 torch Dataset 遍历成 SFT 风格的 JSON，是连接参考 Dataset 与主训练脚本的桥梁。
# 它按 train/val/test 划分导出，每条记录含 historyList / itemList / trueSelection，正好对应 sft.py、DynamicPO.py、inference.py、evaluate_batch.py 读取的字段；仅用于复现数据流程。
# 导入 argparse：解析命令行参数（--dataset / --data_dir / --output_dir / --cans_num），让导出脚本可配置。
import argparse
# 导入 json：用 json.dump 把 records 写成带缩进的 JSON 文件，供主训练脚本按文件读取。
import json
# 导入 os：用 os.makedirs 创建输出目录、os.path.join 拼接输出文件路径。
import os

# 从 tqdm 导入进度条 tqdm：遍历数据集样本时显示进度，便于观察大数据集导出进度。
from tqdm import tqdm

# 导入 GoodreadsData 类：注册到 DATASET_REGISTRY 的 'goodreads' 键，用于导出 Goodreads 图书数据。
from goodreads_data import GoodreadsData
# 导入 LastfmData 类：注册到 'lastfm' 键，用于导出 LastFM 艺术家数据。
from lastfm_data import LastfmData
# 导入 SteamData 类：注册到 'steam' 键，用于导出 Steam 游戏数据。
from steam_data import SteamData


# 定义数据集注册表：把命令行 --dataset 的字符串映射到对应的 Dataset 类，export_dataset 据此实例化，三个数据集共用同一套导出逻辑。
DATASET_REGISTRY = {
    # 'goodreads': GoodreadsData —— 注册表条目，键与 --dataset 取值一致，值指向 Goodreads 的 Dataset 类。
    "goodreads": GoodreadsData,
    # 'lastfm': LastfmData —— 注册表条目，指向 LastFM 的 Dataset 类。
    "lastfm": LastfmData,
    # 'steam': SteamData —— 注册表条目，指向 Steam 的 Dataset 类。
    "steam": SteamData,
# 结束注册表字典；字典最后一个条目后无逗号，符合 Python 语法。
}


# 定义导出函数 export_dataset：dataset_name 选数据集、data_dir 指向预处理 .df、output_dir 为输出目录、cans_num 默认 20（对应 cans20 候选集）。
def export_dataset(dataset_name: str, data_dir: str, output_dir: str, cans_num: int = 20):
    # 用数据集名从注册表取出对应 Dataset 类（GoodreadsData / LastfmData / SteamData），实现按名称分发。
    dataset_cls = DATASET_REGISTRY[dataset_name]
    # 设定要导出的三个划分 train / val / test，对应交互式序列划分的训练/验证/测试，循环逐一生成独立 JSON。
    splits = ["train", "val", "test"]
    # 创建输出目录，exist_ok=True 保证目录已存在时不报错，避免重复创建失败。
    os.makedirs(output_dir, exist_ok=True)

    # 遍历三个划分：每个 split 单独实例化 Dataset 并导出一个 JSON 文件。
    for split in splits:
        # 按当前 split 实例化数据集，传入 data_dir、stage=split、cans_num；内部会按 stage 读取对应 .df 并做去填充/历史拼接/负采样。
        dataset = dataset_cls(data_dir=data_dir, stage=split, cans_num=cans_num)
        # 初始化 records 空列表，累积该划分下每条样本导出的字典。
        records = []
        # 用 tqdm 包裹 range(len(dataset)) 逐条遍历，desc 显示 '数据集名-划分' 便于区分导出进度。
        for i in tqdm(range(len(dataset)), desc=f"{dataset_name}-{split}"):
            # 调用 dataset[i] 取一条样本，触发 __getitem__ 里的负采样，得到含 movie_seq / cans_name / next_title 的 dict。
            sample = dataset[i]
            # 开始构造导出的 JSON 记录字典，字段命名与主训练脚本（sft.py 等）读取的键严格对齐。
            record = {
                # 'historyList': sample['movie_seq'] —— 用户历史拼接结果（'::' 分隔串）；主脚本 sft.py 会用 .split('::') 还原成列表填入 prompt 的 [HistoryHere]。
                "historyList": sample["movie_seq"],
                # 'itemList': sample['cans_name'] —— cans20 候选物品名称列表（19 负样本 + 1 正样本），主脚本填入 prompt 的 [CansHere]，DPO 阶段据此取负样本。
                "itemList": sample["cans_name"],
                # 'trueSelection': sample['next_title'] —— 正样本标题，作为 SFT 的 completion / DPO 的 chosen / 推理评估的标准答案。
                "trueSelection": sample["next_title"],
            # 结束记录字典；三字段构成一条 SFT 训练样本，字段名与主脚本完全一致。
            }
            # 把该条 record 追加进 records，供最终一次性写入 JSON。
            records.append(record)

        # 拼出输出文件路径，命名格式 '<数据集名>-<划分>.json'（如 lastfm-train.json），便于按数据集与划分区分。
        output_path = os.path.join(output_dir, f"{dataset_name}-{split}.json")
        # 以写模式打开输出文件，with 保证写完自动关闭，确保 json.dump 完整落盘。
        with open(output_path, "w") as f:
            # 用 json.dump 把 records 写入文件，indent=4 生成缩进美化、可读的 JSON，供主训练脚本加载。
            json.dump(records, f, indent=4)


# 脚本入口判断：仅当直接运行本文件时执行下面的命令行解析与导出，被 import 时不触发。
if __name__ == "__main__":
    # 创建 ArgumentParser，description 说明用途——从参考数据集生成 SFT 风格 JSON 数据。
    parser = argparse.ArgumentParser(description="Prepare SFT-style JSON data from reference datasets.")
    # 添加 --dataset 参数：choices 取自注册表的键并排序，required=True 强制指定 goodreads/lastfm/steam 之一。
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY.keys()), required=True)
    # 添加必填参数 --data_dir：指向含 id2name 与 *_data.df 的预处理数据目录。
    parser.add_argument("--data_dir", required=True)
    # 添加必填参数 --output_dir：指定导出 JSON 的存放目录。
    parser.add_argument("--output_dir", required=True)
    # 添加 --cans_num 参数：整型，默认 20，控制候选集大小（19 负 + 1 正，即 cans20），传给各 Dataset 的负采样。
    parser.add_argument("--cans_num", type=int, default=20)
    # 解析命令行参数到 args 命名空间，供下方导出调用读取。
    args = parser.parse_args()

    # 调用 export_dataset 开始导出，以下四行逐一传入从命令行解析出的参数。
    export_dataset(
        # dataset_name=args.dataset —— 传入选定的数据集名，用于从注册表取类与命名输出文件。
        dataset_name=args.dataset,
        # data_dir=args.data_dir —— 传入预处理数据目录，供各 Dataset 读取 .df 与 id2name。
        data_dir=args.data_dir,
        # output_dir=args.output_dir —— 传入 JSON 输出目录。
        output_dir=args.output_dir,
        # cans_num=args.cans_num —— 传入候选集规模（默认 20），决定负采样数量与 itemList 长度。
        cans_num=args.cans_num,
    # 结束 export_dataset 调用；最后一个参数后带逗号是合法的，右括号单独成行。
    )
