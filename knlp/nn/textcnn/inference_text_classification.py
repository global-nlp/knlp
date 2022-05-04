# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference_text_classification
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-04-28
# Description:
# -----------------------------------------------------------------------#

import torch
import torch.nn as nn
import json
from knlp.nn.textcnn.dataset_text_classification import TextClassificationDataset
from knlp.nn.bilstm_crf.inference_nn import InferenceNN


class InferenceTextClassification(InferenceNN):
    """
    该类为神经网络文本分类任务推理类
    """

    def __init__(self, model_path: str = "", model: torch.nn.Module = None, word2idx_path: str = "",
                 label2idx_path: str = "", max_length: int = None, tokenizer=None, preprocess_fn=None,
                 device: str = "cpu"):
        """

        Args:
            model_path: 模型路径
            model: 模型
            word2idx_path: word2idx路径
            label2idx_path: label2idx路径
            max_length: token长度
            tokenizer: 分词器
            preprocess_fn: 文本预处理方法
            device: 设备
        """
        assert word2idx_path and max_length
        self.load_word2idx(word2idx_path)
        if label2idx_path:
            self.load_idx2label(label2idx_path)
        else:
            self.idx2label = None
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.preprocess_fn = preprocess_fn
        self.softmax = nn.Softmax(dim=1)
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

    def load_idx2label(self, label2idx_path: str):
        """
        加载id2label
        Args:
            label2idx_path:

        Returns:

        """
        with open(label2idx_path, 'r') as f:
            label2idx = json.load(f)
            self.idx2label = {value: key for key, value in label2idx.items()}

    @staticmethod
    def idx2label_function(labels_idx: list, idx2label: dict):
        """
        idx映射为标签
        Args:
            labels_idx:
            idx2label:

        Returns:

        """
        return [idx2label[label_idx] for label_idx in labels_idx]

    def forward(self, seqs: list, return_label: bool = False):
        """
        序列标注推理运算
        Args:
            seqs:
            return_prob:

        Returns:

        """
        seqs = [TextClassificationDataset.process_text(seq, self.max_length, tokenizer=self.tokenizer,
                                                       preprocess_fn=self.preprocess_fn) for seq in seqs]
        seqs_idx = TextClassificationDataset.seq2idx_function(seqs, self.word2idx)
        labels_prob = self.softmax(self.model(seqs_idx))
        # 返回概率
        if not return_label:
            return labels_prob.tolist()
        # 未载入idx2label
        elif not self.idx2label:
            print("need idx2label!")
            return labels_prob.tolist()
        # 返回label
        else:
            labels_idx = labels_prob.argmax(dim=1).tolist()
            labels = InferenceTextClassification.idx2label_function(labels_idx, self.idx2label)
            return labels

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
