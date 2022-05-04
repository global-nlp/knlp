# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: dataset_text_classification
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-03-29
# Description:
# -----------------------------------------------------------------------#

from knlp.common.constant import UNK, PAD
import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """
    文本分类任务Dataset
    """

    def __init__(self, vocab_set_path: str = None, training_data_path: str = None, eval_data_path: str = None,
                 test_data_path: str = None, mode: str = "train", word2idx: dict = None, label2idx: dict = None,
                 data_delimiter: str = "\t", max_length: int = 100, tokenizer=None, preprocess_fn=None):
        """
        初始化 TextClassificationDataset
        Args:
            vocab_set_path: 词典路径。mode=="train"时有效
            training_data_path: 训练集路径，当mode=="train"时为必要参数。其他mode不影响
            eval_data_path: 验证集路径，当mode=="eval"时为必要参数。其他mode不影响
            test_data_path: 测试集路径，当mode=="test"时为必要参数。其他mode不影响
            mode: Dataset的类型,有"train"、"eval"、"test"三类
            word2idx: 词的索引表，当mode=="eval"或mode=="test"时为必要参数。当mode=="train"时不影响
            label2idx: label索引表，当mode=="eval"时为必要参数。其他mode不影响
            data_delimiter: seqs 与 label之间的分隔符
            max_length: token最大长度
            tokenizer: 分词器
            preprocess_fn: 文本预处理方法
        """
        self.mode = mode
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.data_delimiter = data_delimiter
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.preprocess_fn = preprocess_fn
        # 判断参数是否齐全
        assert (mode == "train" and training_data_path) or \
               (mode == "eval" and eval_data_path and word2idx and label2idx) or \
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
        print("Loading {} data".format(self.mode))
        # 空数据
        if not datas:
            return [], []
        seqs = []
        labels = []
        for line in datas:
            # 去除\n
            line = line.strip()
            line_split = line.split(self.data_delimiter)
            if len(line_split) > 2:
                # 提醒text中含有分隔符
                continue
            # len(line_split)==1，可能是测试集，没写label
            elif len(line_split) == 2:
                seq, label = line_split
            else:
                seq, label = line_split[0], ''
            seq = TextClassificationDataset.process_text(seq, self.max_length, tokenizer=self.tokenizer)
            seqs.append(seq)
            labels.append(label)
        print("{} data has been loaded and processed successfully".format(self.mode))
        return seqs, labels

    @staticmethod
    def process_text(seq: str, max_length: int, tokenizer=None, preprocess_fn=None):
        """
        对seq进行预处理、分词、截断、PAD
        Args:
            seq:
            tokenizer:
            max_length:
            preprocess_fn:

        Returns:

        """
        # 文本预处理
        seq = seq if preprocess_fn == None else preprocess_fn(seq)
        # 分词
        seq = tokenizer(seq) if tokenizer != None else list(seq)
        # 截断、PAD
        seq = TextClassificationDataset.padding(seq, max_length)
        return seq

    @staticmethod
    def padding(seq: list, max_length: int):
        """
        对token进行截断和padding
        Args:
            seq:分词后的seq
            max_length:

        Returns:

        """
        if len(seq) > max_length:
            seq = seq[:max_length]
        else:
            seq = seq + [PAD for _ in range(max_length - len(seq))]
        return seq

    def _get_word2idx_from_vacab_data(self, vocab_data: list):
        """
        根据词典构建索引表
        Args:
            vocab_data:

        Returns:

        """
        word2idx = {UNK: 0, PAD: 1}
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
        word2idx = {UNK: 0, PAD: 1}
        for seq in seqs:
            for word in seq:
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
        return word2idx

    def _get_label2idx(self, labels: list):
        """
        根据labels构建索引表
        Args:
            labels:

        Returns:

        """
        # 去重
        labels = list(set(labels))
        # labels全为数字时,label索引按照label大小排序
        if "".join(labels).isdigit():
            labels = sorted(labels, key=lambda x: int(x))
        label2idx = {}
        for label in labels:
            if label not in label2idx:
                label2idx[label] = len(label2idx)
        return label2idx

    def _read_file(self, path: str = None):
        """
        加载数据
        Args:
            path:

        Returns:

        """
        with open(path) as f:
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
            return torch.tensor([[word2idx.get(word, word2idx[UNK]) for word in seq] for seq in seqs], dtype=torch.long)
        else:
            return [[word2idx.get(word, word2idx[UNK]) for word in seg] for seg in seqs]

    @staticmethod
    def label2idx_function(labels, label2idx, to_tensor=True):
        """
        将标签列表转换为索引列表
        Args:
            labels:
            label2idx:
            to_tensor:

        Returns:

        """
        if to_tensor:
            return torch.tensor([label2idx[label] for label in labels], dtype=torch.long)
        else:
            return [label2idx[label] for label in labels]

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
        # 加载、处理训练集
        seqs, labels = self._split_data(training_data)

        # 获取数据集中的word2idx
        # 有词典
        if vocab_set_path:
            # 读字典文件
            vocab_data = self._read_file(vocab_set_path)
            self.word2idx = self._get_word2idx_from_vacab_data(vocab_data)
        # 无词典
        else:
            self.word2idx = self._get_word2idx_from_seqs(seqs)
        # 获取label2idx
        self.label2idx = self._get_label2idx(labels)

        self.seqs_idx = TextClassificationDataset.seq2idx_function(seqs, self.word2idx, to_tensor=True)
        self.labels_idx = TextClassificationDataset.label2idx_function(labels, self.label2idx, to_tensor=True)

    def _load_eval_data(self, eval_data_path: str):
        """
        mode=="eval"时 加载验证集
        Args:
            eval_data_path:

        Returns:

        """
        # 读验证集文件
        eval_data = self._read_file(eval_data_path)
        # 加载、处理验证集
        seqs, labels = self._split_data(eval_data)

        self.seqs_idx = TextClassificationDataset.seq2idx_function(seqs, self.word2idx, to_tensor=True)
        self.labels_idx = TextClassificationDataset.label2idx_function(labels, self.label2idx, to_tensor=True)

    def _load_test_data(self, test_data_path: str):
        """
        mode=="test"时 加载测试集
        Args:
            test_data_path:

        Returns:

        """
        # 读测试集文件
        test_data = self._read_file(test_data_path)
        # 加载、处理测试集
        seqs, _ = self._split_data(test_data)
        self.seqs_idx = TextClassificationDataset.seq2idx_function(seqs, self.word2idx, to_tensor=True)

    @property
    def vocab_size(self):
        """
        词典大小
        Returns:

        """
        return len(self.word2idx)

    @property
    def labelset_size(self):
        """
        标签数量
        Returns:

        """
        return len(self.label2idx)

    def __getitem__(self, index):
        if self.mode in ["train", "eval"]:
            return self.seqs_idx[index], self.labels_idx[index]
        else:
            return self.seqs_idx[index]

    def __len__(self):
        return len(self.seqs_idx)


if __name__ == "__main__":
    from knlp.common.constant import KNLP_PATH
    from torch.utils.data import DataLoader
    import jieba

    args = {"vocab_set_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_vocab_25k.txt",
            "training_data_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_train_110k.txt",
            "eval_data_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_eval_9988.txt",
            "tokenier": jieba.lcut}
    train_dataset = TextClassificationDataset(**args)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    word2idx, label2idx = train_dataset.word2idx, train_dataset.label2idx
    eval_dataset = TextClassificationDataset(mode="eval", word2idx=word2idx, label2idx=label2idx, **args)
    eval_data_loader = DataLoader(eval_dataset, batch_size=eval_dataset.__len__())

    # for seqs_idx, labels_idx in train_data_loader:
    #     print(seqs_idx)
    #     print(labels_idx)
    #     print('___________')
    # for seqs_idx, labels_idx in eval_data_loader:
    #     print(seqs_idx)
    #     print('___________')
    # print(eval_dataset.label2idx)
    # print(eval_dataset.word2idx)
