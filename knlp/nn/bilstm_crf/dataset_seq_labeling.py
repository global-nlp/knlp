# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: dataset_seq_labeling
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-03-29
# Description:
# -----------------------------------------------------------------------#

from knlp.common.constant import UNK, delimiter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SeqLabelDataset(Dataset):
    """
    序列标注任务Dataset
    """

    def __init__(self, vocab_set_path: str = None, training_data_path: str = None, eval_data_path: str = None,
                 test_data_path: str = None, mode: str = "train", word2idx: dict = None, tag2idx: dict = None):
        """
        初始化 SeqLabelDataset
        Args:
            vocab_set_path: 词典路径，当mode=="train"时有效。其他mode不影响
            training_data_path: 训练集路径，当mode=="train"时为必要参数。其他mode不影响
            eval_data_path: 验证集路径，当mode=="eval"时为必要参数。其他mode不影响
            test_data_path: 测试集路径，当mode=="test"时为必要参数。其他mode不影响
            mode: Dataset的类型,有"train"、"eval"、"test"三类
            word2idx: 词的索引表，当mode=="eval"或mode=="test"时为必要参数。当mode=="train"时不影响
            tag2idx: 标签索引表，当mode=="eval"时为必要参数。其他mode不影响
        """
        self.mode = mode
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        # 判断参数是否齐全
        assert (mode == "train" and training_data_path) or \
               (mode == "eval" and eval_data_path and word2idx and tag2idx) or \
               (mode == "test" and test_data_path and word2idx)
        if mode == "train":
            self._load_train_data(vocab_set_path=vocab_set_path, training_data_path=training_data_path)
        elif mode == "eval":
            self._load_eval_data(eval_data_path=eval_data_path)
        elif mode == "test":
            self._load_test_data(test_data_path=test_data_path)

    def _split_data(self, datas: list):
        """
        将加载的数据分句
        Args:
            datas:

        Returns:

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
                line = line.split(delimiter)
                seq.append(line[0])
                tag.append(line[1])
            # 句末且句子不为空:存储
            elif seq:
                seqs.append(seq)
                tags.append(tag)
                seq = []
                tag = []
        return seqs, tags

    def _get_word2idx_from_vacab_data(self, vocab_data: list):
        """
        根据词典构建索引表
        Args:
            vocab_data:

        Returns:

        """
        word2idx = {UNK: 0}
        for word in vocab_data:
            word = word.strip()
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        return word2idx

    def _get_word2idx_from_seqs(self, seqs: list):
        """
        根据seqs构建索引表
        Args:
            seqs:

        Returns:

        """
        word2idx = {UNK: 0}
        for seq in seqs:
            for word in seq:
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
        return word2idx

    def _get_tag2idx(self, tags: list):
        """
        根据tags构建索引表
        Args:
            tags:

        Returns:

        """
        tag2idx = {}
        for tag in tags:
            for t in tag:
                if t not in tag2idx:
                    tag2idx[t] = len(tag2idx)
        return tag2idx

    def _read_file(self, path: str = None):
        """
        加载数据
        Args:
            path:

        Returns:

        """
        with open(path, encoding='utf-8') as f:
            return f.readlines()

    @staticmethod
    def seq2idx_function(seqs, word2idx, to_tensor=True):
        """
        将字词列表转换为索引列表
        Args:
            seqs:
            word2idx:
            to_tensor:

        Returns:

        """
        if to_tensor:
            return [torch.tensor([word2idx.get(word, word2idx[UNK]) for word in seq], dtype=torch.long) for seq in seqs]
        else:
            return [[word2idx.get(word, word2idx[UNK]) for word in seg] for seg in seqs]

    @staticmethod
    def tag2idx_function(tags, tag2idx, to_tensor=True):
        """
        将标签列表转换为索引列表
        Args:
            tags:
            tag2idx:
            to_tensor:

        Returns:

        """
        if to_tensor:
            return [torch.tensor([tag2idx[t] for t in tag], dtype=torch.long) for tag in tags]
        else:
            return [[tag2idx[t] for t in tag] for tag in tags]

    @staticmethod
    def prepare_pair_data(datas: list, word2idx: dict, tag2idx: dict):
        """
        将 训练集/验证集 按长度排序，转换为索引列表，并转换为tensor
        Args:
            datas:
            word2idx:
            tag2idx:

        Returns:

        """
        seqs, tags = zip(*datas)
        seqs, tags = list(seqs), list(tags)
        # sequence转换为索引并转换为tensor
        seqs_idx = SeqLabelDataset.seq2idx_function(seqs, word2idx, to_tensor=True)
        # 计算所有的文本长度
        lengths = list(map(len, seqs_idx))
        # 文本长度从大到小排序
        lengths, idx_sort = torch.sort(torch.tensor(lengths), descending=True)
        lengths = lengths.tolist()
        # padding成矩阵
        seqs_idx = pad_sequence(seqs_idx, batch_first=True)[idx_sort]
        # tags转换为索引并转换为tensor
        tags_idx = SeqLabelDataset.tag2idx_function(tags, tag2idx, to_tensor=True)
        # padding成矩阵并排序
        tags_idx = pad_sequence(tags_idx, batch_first=True)[idx_sort]

        return seqs_idx, tags_idx, lengths

    @staticmethod
    def prepare_seq_data(datas: list, word2idx: dict):
        """
        将 测试集 按长度排序，转换为索引列表，并转换为tensor
        Args:
            datas:
            word2idx:

        Returns:

        """
        seqs = datas
        seqs_idx = SeqLabelDataset.seq2idx_function(seqs, word2idx, to_tensor=True)
        # 计算所有的文本长度
        lengths = list(map(len, seqs_idx))
        # 文本长度从大到小排序
        lengths, idx_sort = torch.sort(torch.tensor(lengths), descending=True)
        lengths = lengths.tolist()
        # padding成矩阵
        seqs_idx = pad_sequence(seqs_idx, batch_first=True)[idx_sort]
        # idx_sort还原排序
        _, restore_idx_sort = torch.sort(idx_sort, descending=False)
        return seqs_idx, lengths, restore_idx_sort

    def _load_train_data(self, vocab_set_path: str, training_data_path: str):
        """
        mode=="train"时 加载训练集
        Args:
            vocab_set_path:
            training_data_path:

        Returns:

        """

        # 读训练集文件
        training_data = self._read_file(training_data_path)
        # 将训练集分句
        seqs, tags = self._split_data(training_data)
        # 获取word2idx
        if vocab_set_path:
            # 读字典文件
            vocab_data = self._read_file(vocab_set_path)
            self.word2idx = self._get_word2idx_from_vacab_data(vocab_data)
        else:
            self.word2idx = self._get_word2idx_from_seqs(seqs)
        # 获取tag2idx
        self.tag2idx = self._get_tag2idx(tags)
        self.seqs_idx = SeqLabelDataset.seq2idx_function(seqs, self.word2idx, to_tensor=True)
        self.tags_idx = SeqLabelDataset.tag2idx_function(tags, self.tag2idx, to_tensor=True)

    def _load_eval_data(self, eval_data_path: str):
        """
        mode=="eval"时 加载验证集
        Args:
            eval_data_path:

        Returns:

        """
        # 读验证集文件
        eval_data = self._read_file(eval_data_path)
        # 将验证集分句
        seqs, tags = self._split_data(eval_data)
        self.seqs_idx = SeqLabelDataset.seq2idx_function(seqs, self.word2idx, to_tensor=True)
        self.tags_idx = SeqLabelDataset.tag2idx_function(tags, self.tag2idx, to_tensor=True)

    def _load_test_data(self, test_data_path: str):
        """
        mode=="test"时 加载测试集
        Args:
            test_data_path:

        Returns:

        """
        # 读测试集文件
        test_data = self._read_file(test_data_path)
        # 将测试集分句
        seqs, _ = self._split_data(test_data)
        self.seqs_idx = SeqLabelDataset.seq2idx_function(seqs, self.word2idx, to_tensor=True)

    @property
    def vocab_size(self):
        """
        词典大小
        Returns:

        """
        return len(self.word2idx)

    @property
    def tagset_size(self):
        """
        标签数量
        Returns:

        """
        return len(self.tag2idx)

    @staticmethod
    def collate_fn_train(batch):
        seqs_idx, tags_idx = zip(*batch)
        # 计算所有的文本长度
        lengths = list(map(len, seqs_idx))
        # 文本长度从大到小排序
        lengths, idx_sort = torch.sort(torch.tensor(lengths), descending=True)
        lengths = lengths.tolist()
        # padding成矩阵
        seqs_idx = pad_sequence(seqs_idx, batch_first=True)[idx_sort]
        # padding成矩阵并排序
        tags_idx = pad_sequence(tags_idx, batch_first=True)[idx_sort]
        return seqs_idx, tags_idx, lengths

    @staticmethod
    def collate_fn_test(batch):
        seqs_idx = list(batch)
        # 计算所有的文本长度
        lengths = list(map(len, seqs_idx))
        # 文本长度从大到小排序
        lengths, idx_sort = torch.sort(torch.tensor(lengths), descending=True)
        # padding成矩阵
        seqs_idx = pad_sequence(seqs_idx, batch_first=True)[idx_sort]
        return seqs_idx

    def __getitem__(self, index):
        if self.mode in ["train", "eval"]:
            return self.seqs_idx[index], self.tags_idx[index]
        else:
            return self.seqs_idx[index]

    def __len__(self):
        return len(self.seqs_idx)


if __name__ == "__main__":
    from knlp.common.constant import KNLP_PATH
    from torch.utils.data import DataLoader

    args = {"vocab_set_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_vocab.txt",
            "training_data_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_training_data.txt",
            "eval_data_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_test_data.txt",
            "test_data_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_test_data.txt"}
    train_dataset = SeqLabelDataset(**args)
    train_data_loader = DataLoader(train_dataset, batch_size=64, collate_fn=train_dataset.collate_fn_train)
    word2idx, tag2idx = train_dataset.word2idx, train_dataset.tag2idx
    eval_dataset = SeqLabelDataset(mode="eval", word2idx=word2idx, tag2idx=tag2idx, **args)
    eval_data_loader = DataLoader(eval_dataset, batch_size=eval_dataset.__len__(),
                                  collate_fn=eval_dataset.collate_fn_train)
    test_dataset = SeqLabelDataset(mode="test", word2idx=word2idx, **args)
    test_data_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__(),
                                  collate_fn=eval_dataset.collate_fn_test)
    for seqs_idx, tags_idx, lengths in train_data_loader:
        print(seqs_idx)
        print(tags_idx)
        print(lengths)
        print('___________')
    for seqs_idx in test_data_loader:
        print(seqs_idx)
        print('___________')
    print(eval_dataset.tag2idx)
    print(eval_dataset.word2idx)
