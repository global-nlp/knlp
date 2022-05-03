# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference_seq_labeling
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-04-08
# Description:
# -----------------------------------------------------------------------#

import torch
import json
from knlp.nn.bilstm_crf.dataset_seq_labeling import SeqLabelDataset
from knlp.nn.bilstm_crf.inference_nn import InferenceNN


class InferenceSeqLabel(InferenceNN):
    """
    该类为神经网络序列标注任务推理类
    """

    def __init__(self, model_path: str = "", model: torch.nn.Module = None, word2idx_path: str = "",
                 tag2idx_path: str = "", device: str = "cpu"):
        """

        Args:
            model_path: 模型路径
            model: 模型
            word2idx_path: word2idx路径
            idx2tag_path: id2tag路径
            device: 设备
        """
        assert word2idx_path and tag2idx_path
        self.load_word2idx(word2idx_path)
        self.load_idx2tag(tag2idx_path)
        super().__init__(model_path=model_path, model=model, device=device)

    def load_word2idx(self, word2idx_path: str):
        """
        加载word2idx
        Args:
            word2idx_path:

        Returns:

        """
        with open(word2idx_path, 'r') as f:
            self.word2idx = json.load(f)

    def load_idx2tag(self, tag2idx_path: str):
        """
        加载id2tag
        Args:
            tag2idx_path:

        Returns:

        """
        with open(tag2idx_path, 'r') as f:
            tag2idx = json.load(f)
            self.idx2tag = {value: key for key, value in tag2idx.items()}

    @staticmethod
    def idx2tag_function(tags_idx, idx2tag):
        """
        idx映射为标签
        Args:
            tags_idx:
            idx2tag:

        Returns:

        """
        return [[idx2tag[id] for id in ids] for ids in tags_idx]

    def forward(self, input):
        """
        序列标注推理运算
        Args:
            input:

        Returns:

        """
        seqs_idx, lengths, restore_idx_sort = SeqLabelDataset.prepare_seq_data(input, self.word2idx)
        tags_idx = self.model(seqs_idx, lengths)
        tags_idx = InferenceSeqLabel.restore_sort(tags_idx, restore_idx_sort)
        tags = InferenceSeqLabel.idx2tag_function(tags_idx, self.idx2tag)
        return tags

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
