# 文件级说明：lastfm_data.py 是 LastFM 数据集的预处理参考代码，定义 LastfmData(torch Dataset)，把 LLaRA 产出的 .df 交互数据转成带 cans20 候选集的样本。
# 它仅用于复现数据流程（交互式序列划分、S-DPO 负采样、用户历史拼接），并非主训练脚本；导出的字段经 prepare_sft_data.py 对应到 sft.py / DynamicPO.py 的 historyList / itemList / trueSelection。
# 导入 PyTorch：把交互序列 seq、候选集 cans、目标 next_id 转成张量并 .to(device) 搬到指定设备，是 Dataset 返回张量的基础。
import torch
# 导入 os.path 并别名 op：用 op.join 拼接 data_dir 与 id2name.txt / *_data.df 等文件路径，屏蔽不同操作系统路径分隔符差异。
import os.path as op
# 导入 torch.utils.data 并别名 data：让 LastfmData 继承 data.Dataset，从而能被 DataLoader 按 __len__ / __getitem__ 批量取样。
import torch.utils.data as data

# 导入 pandas：用 pd.read_pickle 读取 LLaRA 预处理好的 DataFrame（含 seq / len_seq / next 列），并做列筛选与 apply 映射。
import pandas as pd
# 导入 random：用 random.Random(next_item) 构造以目标物品为种子的独立随机数发生器，保证每个 next_item 的负样本候选可复现。
import random

# 定义 LastfmData 类并继承 torch 的 Dataset：对应 LastFM（音乐艺术家）数据集，padding_item_id=4606，把交互序列封装成可迭代样本；另两个数据集为 GoodreadsData / SteamData。
class LastfmData(data.Dataset):
    # 构造函数签名起始，data_dir 默认指向 LLaRA 预处理数据目录 ./LLara-Data/lastfm（LastFM 沿用 LLaRA 的目录约定，与 Goodreads/Steam 的 ./data/ 不同）。
    def __init__(self, data_dir='./LLara-Data/lastfm',
                 # stage 参数指定数据划分 train / val / test（交互式序列划分的三种切分），决定 check_files 读取哪个 *_data.df 文件；默认 None。
                 stage=None,
                 # cans_num=20 指定候选集大小：19 个负样本 + 1 个正样本，即 README 里 lastfm-sft-cans20.zip 所指的 cans20 候选集，也是 S-DPO 负采样规模。
                 cans_num=20,
                 # sep 默认 ', '，是历史/候选条目在 prompt 中拼接时的分隔符（对应主训练脚本 trainer/utils.py 里 ', '.join 的分隔方式）。
                 sep=", ",
                 # device 指定张量所在设备（cuda/cpu），__getitem__ 里 seq、cans、next_id 等张量都会 .to(self.device)。
                 device=None,
                 # no_augment=True 默认关闭数据增强；与 self.aug=(stage=='train') and not no_augment 配合，仅当训练且允许增强时才启用增强。
                 no_augment=True):
        # 把构造函数所有局部变量（data_dir、stage、cans_num、sep、device、no_augment，甚至 self 本身）写入实例属性，故后续可直接用 self.stage / self.cans_num / self.data_dir，无需逐项赋值。
        self.__dict__.update(locals())
        # 计算增强开关：仅训练集且 no_augment 为 False 时才做增强；默认 True 表示参考实现不启用增强，保证与预处理数据一致。
        self.aug = (stage=='train') and not no_augment
        # 设定 LastFM 的填充 item id 为 4606（数据集物品总数对应），remove_padding 会据此从 seq 中剔除左侧填充；Goodreads 为 4550、Steam 为 3581，三者不同。
        self.padding_item_id = 4606
        # 显式保存 device 到实例：覆盖 locals() 里的同名值，供 __getitem__ 里所有 torch.tensor(...).to(self.device) 使用。
        self.device = device
        # 构建 item id→名称 的映射：LastFM 在 __init__ 里先读 id2name.txt（艺术家名），再交给 check_files；Goodreads/Steam 则把这一步放进 check_files 开头。
        self.item_id2name = self.get_game_id2name()
        # 调用 check_files 按 stage 选择 train/val/test 的 .df 并生成 session_data，完成初始化时的数据加载与预处理。
        self.check_files()

    # 定义读取 id→名称 映射的方法：LastFM 逐行解析 id2name.txt，把整型 id 映射到艺术家名字符串。
    def get_game_id2name(self):
        # 初始化空字典，用于累积 id(int) → name(str) 映射，供后续把 seq / next / 候选集里的 id 转成可读标题。
        game_id2name = dict()
        # 拼出映射文件路径：LastFM 使用 id2name.txt（Goodreads 同名，Steam 用 steam_id2name.txt），位于 data_dir 下。
        item_path = op.join(self.data_dir, 'id2name.txt')
        # 以只读文本方式打开映射文件，用 with 保证读完自动关闭，避免文件句柄泄漏。
        with open(item_path, 'r') as f:
            # 逐行遍历映射文件的每一行，每行形如 'id::name'，对应一个物品（艺术家）的编号与名称。
            for l in f.readlines():
                # 去掉行尾换行后按 '::' 切分：得到 [id 字符串, name 字符串]；与代码中历史序列用 '::' 拼接的分隔符保持一致。
                ll = l.strip('\n').split('::')
                # 把 id 转成整数作键、去除首尾空白的名称作值存入映射；LastFM 直接保留 ll[1]，不像 Goodreads 需要按 '(' 裁掉作者括号。
                game_id2name[int(ll[0])] = ll[1].strip()
        # 返回构建好的 id→名称 映射，赋给 self.item_id2name，供 seq_to_title、id_to_movie、cans_name 等处把 id 转成标题。
        return game_id2name

    # 定义按 stage 选择数据文件并加载的方法，是交互式序列划分 train/val/test 的入口。
    def check_files(self):
        # 训练阶段：选用训练集 .df，包含用于 SFT/偏好优化的交互序列与目标。
        if self.stage=='train':
            # 训练划分对应文件名 train_data.df（DataFrame pickle），三个数据集的训练/验证文件名一致。
            filename="train_data.df"
        # 验证阶段：选用验证集，用于训练过程中评估，不参与梯度更新。
        elif self.stage=='val':
            # 验证划分对应文件名 val_data.df。
            filename="val_data.df"
        # 测试阶段：选用测试集，对应最终评估 evaluate_batch.py 使用的数据。
        elif self.stage=='test':
            # 测试划分：LastFM 用小写 test_data.df；Goodreads / Steam 用首字母大写的 Test_data.df，是三数据集的字段/文件命名差异之一。
            filename="test_data.df"
        # 兜底分支：stage 不是 train/val/test 时进入，防止读取不存在的文件。
        else:
            # 抛出带具体 stage 值的异常，明确拒绝非法划分名，避免静默读错文件。
            raise ValueError(f"Unsupported stage: {self.stage}")
        # 把 data_dir 与选定的 filename 拼成完整 .df 路径。
        data_path=op.join(self.data_dir, filename)
        # 加载并预处理该划分的数据，得到含 seq_unpad / movie_seq / next_title 等列的 DataFrame，存为 self.session_data 供 __len__ / __getitem__ 使用。
        self.session_data = self.session_data4frame(data_path, self.item_id2name)

    # 定义 DataFrame 预处理主流程：读入 .df、过滤短序列、去填充、把 id 序列拼成标题串，是用户历史拼接与标签构造的核心；game_id2name 为 id→名称 映射。
    def session_data4frame(self, datapath, game_id2name):
        # 用 pd.read_pickle 读取 LLaRA 预处理好的 DataFrame，每行含 seq（含 padding 的交互序列）、len_seq（真实长度）、next（下一个交互，即正样本）。
        train_data = pd.read_pickle(datapath)
        # 交互序列划分过滤：只保留真实交互数 ≥3 的样本，保证有足够历史构造推荐上下文，过短序列无法形成有效用户兴趣。
        train_data = train_data[train_data['len_seq'] >= 3]
        
        # 定义去填充函数：把 seq 中用于对齐的 padding_item_id 剔除，得到真实交互序列 seq_unpad。
        def remove_padding(xx):
            # 用切片复制一份列表再操作，避免就地修改 DataFrame 里原始的 seq，防止副作用污染后续逻辑。
            x = xx[:]
            # 最多尝试 10 次删除：序列左填充可能有多个 padding 标记，循环逐个清除；10 是经验上限。
            for i in range(10):
                # 用 try 包裹 remove：当列表里已无该填充值时 .remove 会抛 ValueError，需要被捕获。
                try:
                    # 从副本中删除一个 padding_item_id（LastFM 为 4606），逐步还原真实交互序列；每轮只删一个，靠循环清干净。
                    x.remove(self.padding_item_id)
                # 捕获 .remove 抛出的异常：说明填充已删尽，此为预期的正常终止信号。
                except:
                    # 填充删尽后立即跳出循环，避免无意义的空转与异常，得到干净的真实序列。
                    break
            # 返回去掉填充后的真实交互 id 列表，作为该行 seq_unpad 的值。
            return x
            
        # 对每行 seq 应用 remove_padding，生成 seq_unpad 列：真实交互 id 序列，用于负采样时排除已交互物品、以及拼接用户历史。
        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
        
        # 定义把 id 序列转成标题串的函数，实现用户历史拼接：将交互历史由 id 列表变为可读文本。
        def seq_to_title(x):
            # 把序列里每个 item id 经 item_id2name 映射成名称；LastFM 此处用 self.item_id2name（Steam 则用传入的局部 game_id2name）。
            game_list = [self.item_id2name[i] for i in x]
            # 用 '::' 把历史名称拼成单个字符串：对应主训练脚本 sft.py 里 d['historyList'].split('::') 的逆操作，导出后即 historyList 字段。
            title_str = '::'.join(game_list)
            # 返回 '::' 分隔的历史标题串，写入 movie_seq 列供下游作为 prompt 里的用户历史。
            return title_str
            
        # 对 seq_unpad 应用 seq_to_title，生成 movie_seq 列——即用户历史拼接结果，后续映射为 JSON 的 historyList。
        train_data['movie_seq'] = train_data['seq_unpad'].apply(seq_to_title)
        
        # 定义把单个目标 item id 转成标题的函数，用于把正样本 next 变成可读名称。
        def id_to_movie(x):
            # 按 id 查表返回名称；LastFM 用 self.item_id2name，Goodreads/Steam 用各自的映射（Steam 用 game_id2name）。
            return self.item_id2name[x]
            
        # 把正样本 next 的 id 映射成标题，生成 next_title 列：即 SFT 的 completion / DPO 的 chosen，导出后对应 trueSelection。
        train_data['next_title'] = train_data['next'].apply(id_to_movie)
        # 返回增广后的 DataFrame（含 seq_unpad / movie_seq / next_title），赋给 self.session_data，完成该划分的预处理。
        return train_data

    # 定义样本数，供 DataLoader / tqdm 遍历；LastFM 用 len(self.session_data)，Goodreads 用 len(self.session_data['seq'])、Steam 用 len(self.session_data)。
    def __len__(self):
        # 返回 DataFrame 行数，即该划分下去掉短序列后的样本总数。
        return len(self.session_data)

    # 定义按下标取一条样本：构造含历史、cans20 候选集、正样本的完整 dict，是 SFT / DPO 取数时的直接接口。
    def __getitem__(self, i):
        # 用 iloc 取第 i 行，得到含 seq / len_seq / next / movie_seq / next_title 等字段的 Series。
        temp = self.session_data.iloc[i]
        # 以真实交互序列和目标 next 调用负采样，得到 cans_num=20 个候选 id（19 负 + 1 正），构成 cans20 候选集。
        candidates = self.negative_sampling(temp['seq_unpad'],temp['next'])
        # 把候选 id 列表映射成名称列表，作为候选物品文本，导出后对应 JSON 的 itemList。
        cans_name=[self.item_id2name[can] for can in candidates]
        # 把含 padding 的原始交互序列转成张量并搬到 device，保留 LLaRA 的对齐形式供可能的 id 级模型使用。
        seq = torch.tensor(temp['seq']).to(self.device)
        # 把真实序列长度转成张量搬到 device，便于模型按长度做 mask 或截断。
        len_seq = torch.tensor(temp['len_seq']).to(self.device)
        # 额外保留 python 原生 int 形式的序列长度，供不需张量的分支（如切片、统计）直接使用。
        len_seq_list = temp['len_seq']
        
        # 开始构造返回样本字典：把历史文本、候选集、正样本等汇总成一条训练/评估样本。
        sample = {
            # 'seq': seq —— 放入张量化的交互序列 seq（含 padding）。
            'seq': seq,
            # 'movie_seq': temp['movie_seq'] —— 放入 '::' 分隔的用户历史标题串，对应主训练脚本的 historyList。
            'movie_seq': temp['movie_seq'],
            # 'len_seq': len_seq —— 放入张量化的真实序列长度。
            'len_seq': len_seq,
            # 'len_seq_list': len_seq_list —— 放入原生 int 形式的序列长度，与 len_seq 并存以适配不同取数方式。
            'len_seq_list': len_seq_list,
            # 'cans': torch.tensor(candidates).to(self.device) —— 把 cans20 候选 id 列表转成张量搬到 device，供 id 级打分 / 评估使用。
            'cans': torch.tensor(candidates).to(self.device),
            # 'cans_name': cans_name —— 放入候选物品名称列表，导出后对应 JSON 的 itemList（含 1 个正样本混入 19 个负样本）。
            'cans_name': cans_name,
            # 'cans_str': '::'.join(cans_name) —— 把候选名称再用 '::' 拼成串，提供与 movie_seq 同风格的文本形式，便于统一按 '::' split 解析。
            'cans_str': '::'.join(cans_name),
            # 'len_cans': self.cans_num —— 记录候选集大小（默认 20），让下游知道 cans / cans_name 的长度，对应 cans20。
            'len_cans': self.cans_num,
            # 'next_id': torch.tensor(temp['next']).to(self.device) —— 把正样本（下一个交互）id 转成张量搬到 device，是序列推荐要预测的目标。
            'next_id': torch.tensor(temp['next']).to(self.device),
            # 'next_title': temp['next_title'] —— 放入正样本标题，供 SFT 的 completion / DPO 的 chosen 使用。
            'next_title': temp['next_title'],
            # 'correct_answer': temp['next_title'] —— 以 correct_answer 再存一份正样本标题作为标准答案：导出时映射为 trueSelection，评估/推理据此判定命中。
            'correct_answer': temp['next_title']
        # 结束样本字典构造；字典最后一个元素后无逗号，符合 Python 语法。
        }
        # 返回该条完整样本，供 prepare_sft_data.py 遍历导出，或供 DataLoader 组批。
        return sample

    # 定义 S-DPO 风格的负采样：从全量物品里排除历史与正样本，采出固定数量负样本拼成 cans20 候选集。
    def negative_sampling(self,seq_unpad,next_item):
        # 候选池 = 全部 item id 去掉已交互历史 seq_unpad、再去掉正样本 next_item；避免把看过物品或答案本身当作负样本。
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        # 用 next_item 作随机种子创建独立 Random：使同一目标物品每次得到完全相同的负样本，保证 SFT/DPO 数据与评估可复现。
        rng = random.Random(next_item)
        # 从候选池无放回采 cans_num-1 个负样本（默认 19），再把正样本 next_item 追加，凑齐 20 个候选，即 cans20。
        candidates=rng.sample(canset, self.cans_num-1)+[next_item]
        # 打乱候选顺序，让正样本不固定在末位，防止模型靠位置捷径而非内容判断，也避免负样本偏置。
        rng.shuffle(candidates)
        # 返回打乱后的 20 个候选 id 列表，__getitem__ 再转成 cans / cans_name / cans_str。
        return candidates
