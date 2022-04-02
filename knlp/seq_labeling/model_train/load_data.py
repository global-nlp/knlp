# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: base_nn_model
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-03-29
# Description:
# -----------------------------------------------------------------------#

import os
from knlp.common.constant import UNK
import random
import torch
from torch.nn.utils.rnn import pad_sequence


class DataLoader:

    def __init__(self, vocab_set_path: str = None, training_data_path: str = None, eval_data_path: str = None,
                 test_data_path: str = None, mode: str = "train", seed: int = 2022, shuffle: bool = True,
                 drop_last: bool = False, batch_size: int = 64, word2idx: dict = None, tag2idx: dict = None):
        """
        初始化 DataLoader
        :param vocab_set_path: 词典路径，当mode=="train"时为必要参数。其他mode不影响
        :param training_data_path: 训练集路径，当mode=="train"时为必要参数。其他mode不影响
        :param eval_data_path: 验证集路径，当mode=="eval"时为必要参数。其他mode不影响
        :param test_data_path: 测试集路径，当mode=="test"时为必要参数。其他mode不影响
        :param mode: DataLoader的类型,有"train"、"eval"、"test"三类
        :param seed: 随机数种子
        :param shuffle: 抽样前是否打乱数据
        :param drop_last: 最后一个batch数据不满batch_size时，是否丢掉最后一个batch
        :param batch_size: 每次抽样的数量
        :param word2idx: 词的索引表，当mode=="eval"或mode=="test"时为必要参数。当mode=="train"时不影响
        :param tag2idx: 标签索引表，当mode=="eval"时为必要参数。其他mode不影响
        """
        self.mode = mode
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.set_seed(seed)
        # 判断参数是否齐全
        assert (mode == "train" and vocab_set_path and training_data_path) or \
               (mode == "eval" and eval_data_path and word2idx and tag2idx) or \
               (mode == "test" and test_data_path and word2idx)
        if mode == "train":
            self._load_train_data(vocab_set_path=vocab_set_path, training_data_path=training_data_path)
        elif mode == "eval":
            self._load_eval_data(eval_data_path=eval_data_path)
        elif mode == "test":
            pass

    def set_seed(self, seed: int):
        """
        设置随机数种子
        :param seed:
        :return:
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _read_file(self, path: str = None):
        """
        加载数据
        :param path:
        :return:
        """
        with open(path) as f:
            return f.readlines()

    def _split_data(self, datas: list):
        """
        将加载的数据分句
        :return: seqs:list 句列表, tags:list 标签列表
        """
        # 空数据
        if not datas:
            return [], []
        # 最后一条数据与其他数据保持一致
        if len(datas) > 0 and datas[-1] != '\n':
            datas.append('\n')
        seqs = []
        tags = []
        seq = []
        tag = []
        for line in datas:
            # 去除\n
            line = line.strip()
            # 句中
            if line:
                line = line.split('\t')
                seq.append(line[0])
                tag.append(line[1])
            # 句末且句子不为空:存储
            elif seq:
                seqs.append(seq)
                tags.append(tag)
                seq = []
                tag = []
        return seqs, tags

    def _get_word2idx(self, vocab_data: list):
        """
        根据词典构建索引表
        :return:
        """
        word2idx = {UNK: 0}
        for word in vocab_data:
            word = word.strip()
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        return word2idx

    def _get_tag2idx(self, tags: list):
        """
        根据tags构建索引表
        :param tags:
        :return:
        """
        tag2idx = {}
        for tag in tags:
            for t in tag:
                if t not in tag2idx:
                    tag2idx[t] = len(tag2idx)
        return tag2idx

    def _shuffle(self):
        """
        将数据打乱
        :return:
        """
        if self.mode == "train" and self.shuffle:
            random.shuffle(self.idx_datas)

    def _seq2idx(self, seqs):
        """
        将字词列表转换为索引列表
        :param seqs:
        :return:
        """
        return [[self.word2idx.get(word, UNK) for word in seg] for seg in seqs]

    def _tag2idx(self, tags):
        """
        将标签列表转换为索引列表
        :param tags:
        :return:
        """
        return [[self.tag2idx.get(t) for t in tag] for tag in tags]

    def _prepare_data(self, datas: list):
        """
        将 训练集/验证集 按长度排序，转换为索引列表，并转换为tensor
        :param datas:
        :return:
        """
        seqs, tags = zip(*datas)
        seqs, tags = list(seqs), list(tags)
        # sequence转换为索引并转换为tensor
        seqs_idx = [torch.tensor([self.word2idx.get(word, self.word2idx[UNK]) for word in seq], dtype=torch.long)
                    for seq in seqs]
        # 计算所有的文本长度
        lengths = list(map(len, seqs_idx))
        # 文本长度从大到小排序
        lengths, idx_sort = torch.sort(torch.tensor(lengths), descending=True)
        lengths = lengths.tolist()
        # padding成矩阵
        seqs_idx = pad_sequence(seqs_idx, batch_first=True)[idx_sort]
        # tags转换为索引并转换为tensor
        tags_idx = [torch.tensor([self.tag2idx[t] for t in tag], dtype=torch.long) for tag in tags]
        # padding成矩阵并排序
        tags_idx = pad_sequence(tags_idx, batch_first=True)[idx_sort]

        return seqs_idx, tags_idx, lengths

    def _load_train_data(self, vocab_set_path: str, training_data_path: str):
        """
        mode=="train"时 加载训练集
        :param vocab_set_path:
        :param training_data_path:
        :return:
        """
        # 读字典文件
        vocab_data = self._read_file(vocab_set_path)
        # 读训练集文件
        training_data = self._read_file(training_data_path)
        # 将训练集分句
        seqs, tags = self._split_data(training_data)
        # 获取word2idx，tag2idx
        self.word2idx = self._get_word2idx(vocab_data)
        self.tag2idx = self._get_tag2idx(tags)
        # seq_idxs = self._seq2idx(seqs)
        # tag_idxs = self._tag2idx(tags)
        self.idx_datas = list(zip(seqs, tags))

    def _load_eval_data(self, eval_data_path: str):
        """
        mode=="eval"时 加载验证集
        :param eval_data_path:
        :return:
        """
        # 读验证集文件
        eval_data = self._read_file(eval_data_path)
        # 将验证集分句
        seqs, tags = self._split_data(eval_data)
        self.idx_datas = list(zip(seqs, tags))
        # 验证集的一次性获取所有
        self.batch_size = self.length

    @property
    def vocab_size(self):
        """
        词典大小
        :return:
        """
        return len(self.word2idx)

    @property
    def tagset_size(self):
        """
        标签数量
        :return:
        """
        return len(self.tag2idx)

    @property
    def length(self):
        """
        数据量大小
        :return:
        """
        return len(self.idx_datas)

    @property
    def _iter_num(self):
        """
        抽样次数
        :return:
        """
        iter_num = self.length // self.batch_size
        if not self.drop_last and self.length % self.batch_size != 0:
            iter_num += 1
        return iter_num

    @property
    def datas(self):
        """
        迭代器：获取数据
        :return:
        """
        self._shuffle()
        for batch in range(self._iter_num):
            start_index = batch * self.batch_size
            end_index = min(start_index + self.batch_size, self.length)
            batch_datas = self.idx_datas[start_index:end_index]
            if self.mode in ["train", "eval"]:
                yield self._prepare_data(batch_datas)


if __name__ == "__main__":
    from knlp.common.constant import KNLP_PATH

    args = {"vocab_set_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_vocab.txt",
            "training_data_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_training_data.txt",
            "eval_data_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_test_data.txt"}
    train_data_loader = DataLoader(**args)
    word2idx, tag2idx = train_data_loader.word2idx, train_data_loader.tag2idx
    eval_data_loader = DataLoader(mode="eval", word2idx=word2idx, tag2idx=tag2idx, **args)

    for seqs_idx, tags_idx, lengths in eval_data_loader.datas:
        print(seqs_idx)
        print(tags_idx)
        print(lengths)
        print('___________')
    print(eval_data_loader.tag2idx)
    print(eval_data_loader.word2idx)
