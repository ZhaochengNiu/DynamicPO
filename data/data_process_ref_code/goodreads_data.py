# 文件级说明：goodreads_data.py 是 Goodreads（图书）数据集的预处理参考代码，定义 GoodreadsData(torch Dataset)，把 .df 交互数据转成带 cans20 候选集的样本。
# 它仅复现数据流程（交互式序列划分、S-DPO 负采样、用户历史拼接），非主训练脚本；字段经 prepare_sft_data.py 导出后对应 sft.py / DynamicPO.py 的 historyList / itemList / trueSelection。
# 导入 PyTorch：把 seq、候选集、next_id 转成张量并搬到 device，供 Dataset 返回张量。
import torch
# 导入 os.path 别名 op：用 op.join 拼 data_dir 与 id2name.txt / *_data.df 路径，屏蔽操作系统分隔符差异。
import os.path as op
# 导入 torch.utils.data 别名 data：让 GoodreadsData 继承 data.Dataset，供 DataLoader 按 __len__ / __getitem__ 批量取样。
import torch.utils.data as data

# 导入 pandas：用 pd.read_pickle 读 LLaRA 预处理 DataFrame（seq / len_seq / next 列）并做筛选与 apply 映射。
import pandas as pd
# 导入 random：用 random.Random(next_item) 以目标物品为种子，保证同一正样本的负样本候选可复现。
import random

# 定义 GoodreadsData 类并继承 Dataset：对应 Goodreads 图书数据集，padding_item_id=4550，书名含作者括号需裁剪；另两个为 LastfmData / SteamData。
class GoodreadsData(data.Dataset):
    # 构造函数签名起始，data_dir 默认 ./data/goodreads（Goodreads/Steam 统一放在 ./data/ 下，与 LastFM 的 ./LLara-Data/lastfm 目录约定不同）。
    def __init__(self, data_dir='./data/goodreads',
                 # stage 指定交互式序列划分 train / val / test，决定 check_files 读取哪个 .df；默认 None。
                 stage=None,
                 # cans_num=20 指定候选集规模：19 负样本 + 1 正样本，即 cans20（对应 README 的数据 zip 与 S-DPO 负采样规模）。
                 cans_num=20,
                 # sep 默认 ', '，是历史/候选在 prompt 中拼接的分隔符，对应 trainer/utils.py 里 ', '.join 的用法。
                 sep=", ",
                 # device 指定张量设备（cuda/cpu），__getitem__ 里 seq、cans、next_id 等张量均 .to(self.device)。
                 device=None,
                 # no_augment=True 默认关闭增强；与 self.aug=(stage=='train') and not no_augment 配合，仅训练且允许时才启用。
                 no_augment=True):
        # 把构造函数局部变量（data_dir、stage、cans_num、sep、device、no_augment、self）写入实例属性，故可直接用 self.stage / self.data_dir 等。
        self.__dict__.update(locals())
        # 计算增强开关：仅训练集且 no_augment 为 False 时增强；默认 True 表示参考实现不启用，保持与预处理数据一致。
        self.aug = (stage=='train') and not no_augment
        # 设定 Goodreads 填充 item id 为 4550，remove_padding 据此剔除 seq 左填充；LastFM 为 4606、Steam 为 3581，三者取值不同。
        self.padding_item_id=4550
        # 显式保存 device 到实例，供 __getitem__ 里所有 torch.tensor(...).to(self.device) 调用。
        self.device=device
        # 显式把 cans_num 存为 self.cans_num（LastFM/Steam 靠 locals() 隐式带出），供 negative_sampling 与 len_cans 读取。
        self.cans_num = cans_num
        # 调用 check_files 按 stage 选 train/val/test 的 .df 并生成 session_data，完成初始化时的数据加载与预处理。
        self.check_files()

    # 定义样本数方法，供 DataLoader / tqdm 遍历；Goodreads 用 len(self.session_data['seq'])，与 LastFM/Steam 的 len(self.session_data) 不同。
    def __len__(self):
        # 返回 seq 列的行数，即该划分下去掉短序列后的样本总数。
        return len(self.session_data['seq'])

    # 定义按下标取一条样本：构造含历史、cans20 候选集、正样本的完整 dict，是 SFT / DPO 取数接口。
    def __getitem__(self, i):
        # 用 iloc 取第 i 行，得到含 seq / len_seq / next / movie_seq / next_title 的 Series。
        temp = self.session_data.iloc[i]
        # 以 seq_unpad 与目标 next 调用负采样，得到 20 个候选 id（19 负 + 1 正），构成 cans20 候选集。
        candidates = self.negative_sampling(temp['seq_unpad'],temp['next'])
        # 把候选 id 映射成书名列表，导出后对应 JSON 的 itemList。
        cans_name=[self.item_id2name[can] for can in candidates]
        # 把含 padding 的原始交互序列转成张量并搬到 device，保留 LLaRA 对齐形式。
        seq = torch.tensor(temp['seq']).to(self.device)
        # 把真实序列长度转成张量搬到 device，供模型做 mask / 截断。
        len_seq = torch.tensor(temp['len_seq']).to(self.device)
        # 额外保留 python 原生 int 形式的序列长度，供不需张量的分支使用。
        len_seq_list = temp['len_seq']
        # 开始构造返回样本字典，汇总历史文本、候选集、正样本等成一条训练/评估样本。
        sample = {
            # 'seq': seq —— 放入张量化的含 padding 交互序列。
            'seq': seq,
            # 'movie_seq': temp['movie_seq'] —— 放入 '::' 分隔的用户历史书名串，对应主训练脚本 historyList。
            'movie_seq': temp['movie_seq'],
            # 'len_seq': len_seq —— 放入张量化的真实序列长度。
            'len_seq': len_seq,
            # 'len_seq_list': len_seq_list —— 放入原生 int 形式的序列长度，适配不同取数方式。
            'len_seq_list': len_seq_list,
            # 'cans': torch.tensor(candidates).to(self.device) —— 把 cans20 候选 id 转张量搬到 device，供 id 级打分/评估。
            'cans': torch.tensor(candidates).to(self.device),
            # 'cans_name': cans_name —— 放入候选书名列表，导出后对应 JSON 的 itemList（正样本混入 19 个负样本）。
            'cans_name': cans_name,
            # 'cans_str': '::'.join(cans_name) —— 把候选书名用 '::' 拼成串，与 movie_seq 同风格，便于统一 split 解析。
            'cans_str': '::'.join(cans_name),
            # 'len_cans': self.cans_num —— 记录候选集大小（默认 20），对应 cans20。
            'len_cans': self.cans_num,
            # 'next_id': torch.tensor(temp['next']).to(self.device) —— 把正样本 id 转张量搬到 device，是要预测的目标。
            'next_id': torch.tensor(temp['next']).to(self.device),
            # 'next_title': temp['next_title'] —— 放入正样本书名，供 SFT completion / DPO chosen 使用。
            'next_title': temp['next_title'],
            # 'correct_answer': temp['next_title'] —— 再存一份正样本书名作标准答案，导出映射为 trueSelection，评估据此判命中。
            'correct_answer': temp['next_title']
        # 结束样本字典构造；字典最后一个元素后无逗号，符合 Python 语法。
        }
        # 返回该条完整样本，供 prepare_sft_data.py 遍历导出或 DataLoader 组批。
        return sample
    
    # 定义 S-DPO 风格负采样：从全量物品排除历史与正样本，采固定数量负样本拼成 cans20 候选集。
    def negative_sampling(self,seq_unpad,next_item):
        # 候选池 = 全部 item id 去掉已交互历史 seq_unpad、再去掉正样本 next_item，避免把读过书或答案本身当负样本。
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        # 用 next_item 作种子创建独立 Random，使同一目标书每次得到相同负样本，保证 SFT/DPO 数据与评估可复现。
        rng = random.Random(next_item)
        # 从候选池无放回采 cans_num-1 个负样本（默认 19），再追加正样本 next_item，凑齐 20 个候选即 cans20。
        candidates=rng.sample(canset, self.cans_num-1)+[next_item]
        # 打乱候选顺序，让正样本不固定在末位，避免模型靠位置捷径判断，也避免负样本偏置。
        rng.shuffle(candidates)
        # 返回打乱后的 20 个候选 id 列表，__getitem__ 再转成 cans / cans_name / cans_str。
        return candidates  

    # 定义按 stage 选数据文件并加载的方法，是交互式序列划分 train/val/test 的入口；Goodreads 在此方法开头先构建 item_id2name。
    def check_files(self):
        # 先构建 id→书名 映射并存入 self.item_id2name（LastFM 在 __init__ 里做），供后续 session_data4frame 与 cans_name 使用。
        self.item_id2name=self.get_game_id2name()
        # 训练阶段：选用训练集 .df，包含用于 SFT/偏好优化的交互序列与目标。
        if self.stage=='train':
            # 训练划分对应文件名 train_data.df（DataFrame pickle），三数据集训练/验证文件名一致。
            filename="train_data.df"
        # 验证阶段：选用验证集，用于训练中评估，不参与梯度更新。
        elif self.stage=='val':
            # 验证划分对应文件名 val_data.df。
            filename="val_data.df"
        # 测试阶段：选用测试集，对应最终评估 evaluate_batch.py 使用的数据。
        elif self.stage=='test':
            # 测试划分：Goodreads 用首字母大写 Test_data.df（Steam 同，LastFM 用小写 test_data.df），是三数据集文件命名差异之一。
            filename="Test_data.df"
        # 兜底分支：stage 非 train/val/test 时进入，防止读取不存在的文件。
        else:
            # 抛出带具体 stage 值的异常，明确拒绝非法划分名，避免静默读错文件。
            raise ValueError(f"Unsupported stage: {self.stage}")
        # 把 data_dir 与选定 filename 拼成完整 .df 路径。
        data_path=op.join(self.data_dir, filename)
        # 加载并预处理该划分数据，得到含 seq_unpad / movie_seq / next_title 等列的 DataFrame，存为 self.session_data 供 __len__ / __getitem__ 使用。
        self.session_data = self.session_data4frame(data_path, self.item_id2name)  

        
    # 定义读取 id→书名 映射的方法：逐行解析 id2name.txt，因书名带 '(作者)' 需按 '(' 裁切后取前半。
    def get_game_id2name(self):
        # 初始化空字典，累积 id(int) → 书名(str) 映射，供把 seq / next / 候选 id 转成可读标题。
        game_id2name = dict()
        # 拼出映射文件路径：Goodreads 用 id2name.txt（LastFM 同名，Steam 用 steam_id2name.txt），位于 data_dir 下。
        item_path=op.join(self.data_dir, 'id2name.txt')
        # 以只读文本方式打开映射文件，with 保证读完自动关闭，避免句柄泄漏。
        with open(item_path, 'r') as f:
            # 逐行遍历映射文件，每行形如 'id::书名(作者)'，对应一本书的编号与名称。
            for l in f.readlines():
                # 去掉行尾换行后按 '::' 切分，得到 [id 字符串, 名称字符串]；与历史序列用 '::' 拼接的分隔符一致。
                ll = l.strip('\n').split('::')
                # 按 '(' 再切分名称：Goodreads 书名后常附 '(作者/年份)'，取第一段才是纯书名（LastFM/Steam 无此步骤，直接 strip）。
                g_name = ll[1].split('(')
                # 把 id 转整数作键、裁掉括号内容并 strip 后的纯书名作值存入映射，保证 prompt 里历史/候选只显示书名。
                game_id2name[int(ll[0])] = g_name[0].strip()
        # 返回构建好的 id→书名 映射，赋给 self.item_id2name，供 seq_to_title / id_to_movie / cans_name 查表。
        return game_id2name
    
    # 定义 DataFrame 预处理主流程：读入 .df、过滤短序列、去填充、把 id 序列拼成书名串，是用户历史拼接与标签构造的核心。
    def session_data4frame(self, datapath, game_id2name):
        # 用 pd.read_pickle 读 LLaRA 预处理 DataFrame，每行含 seq（含 padding 交互序列）、len_seq、next（正样本）。
        train_data = pd.read_pickle(datapath)
        # 交互序列划分过滤：只保留真实交互数 ≥3 的样本，保证有足够历史构造推荐上下文。
        train_data = train_data[train_data['len_seq'] >= 3]
        # 定义去填充函数：把 seq 中用于对齐的 padding_item_id（Goodreads 为 4550）剔除，得到真实交互序列。
        def remove_padding(xx):
            # 用切片复制列表再操作，避免就地修改 DataFrame 里原始 seq 造成副作用。
            x = xx[:]
            # 最多尝试 10 次删除：左填充可能有多个 padding 标记，循环逐个清除。
            for i in range(10):
                # 用 try 包裹 remove：列表里已无该填充值时 .remove 会抛 ValueError，需要捕获。
                try:
                    # 从副本删除一个 padding_item_id（4550），逐步还原真实交互序列。
                    x.remove(self.padding_item_id)
                # 捕获 .remove 的异常：说明填充已删尽，是预期的正常终止信号。
                except:
                    # 填充删尽后立即跳出循环，得到干净的真实序列。
                    break
            # 返回去掉填充后的真实交互 id 列表，作为该行 seq_unpad 的值。
            return x
        # 对每行 seq 应用 remove_padding 生成 seq_unpad 列：真实交互 id 序列，供负采样排除已读物品与拼接历史。
        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)

        # 定义把 id 序列转成书名串的函数，实现用户历史拼接：把交互历史由 id 列表变为可读文本。
        def seq_to_title(x): 
            # 把序列里每个 item id 经 self.item_id2name 映射成书名（Goodreads 用 self.item_id2name，Steam 用传入的 game_id2name）。
            game_list = [self.item_id2name[i] for i in x]
            # 用 '::' 把历史书名拼成单个字符串，对应 sft.py 里 d['historyList'].split('::') 的逆操作，导出后即 historyList。
            title_str = '::'.join(game_list)
            # 返回 '::' 分隔的历史书名串，写入 movie_seq 列供下游作 prompt 里的用户历史。
            return title_str
        
        # 对 seq_unpad 应用 seq_to_title 生成 movie_seq 列——用户历史拼接结果，后续映射为 JSON 的 historyList。
        train_data['movie_seq'] = train_data['seq_unpad'].apply(seq_to_title)


        # 定义把单个目标 item id 转成书名的函数，用于把正样本 next 变成可读名称。
        def id_to_movie(x):
            # 按 id 查 self.item_id2name 返回书名；Goodreads/LastFM 用 self.item_id2name，Steam 用 game_id2name。
            return self.item_id2name[x]
        # 把正样本 next 的 id 映射成书名，生成 next_title 列：即 SFT completion / DPO chosen，导出后对应 trueSelection。
        train_data['next_title'] = train_data['next'].apply(id_to_movie)
        # 返回增广后的 DataFrame（含 seq_unpad / movie_seq / next_title），赋给 self.session_data，完成该划分预处理。
        return train_data




    
