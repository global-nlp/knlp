# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: textrnn
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-05-04
# Description:
# -----------------------------------------------------------------------#

from knlp.nn.base_nn_model import BaseNNModel
import torch
import torch.nn as nn


class TextRNN(BaseNNModel):

    def __init__(self, vocab_size: int, label_size: int, pad_idx: int, embedding_dim: int = 64, hidden_dim: int = 64,
                 bi_rnn_num_layers: int = 1, uni_rnn_num_layers: int = 1, vectors: torch.tensor = None, fine_tune=False,
                 dropout_1: float = 0.5, dropout_2: float = 0.5, device=torch.device("cpu")):
        """
        初始化TextRNN模型
        Args:
            vocab_size: 词典的维度
            label_size: 类别的维度
            pad_idx: embedding中PAD的索引
            embedding_dim: 词向量的维度
            hidden_dim: 隐藏层的维度
            bi_rnn_num_layers: 双向RNN的层数
            uni_rnn_num_layers: 单向RNN的层数
            vectors: 预训练词向量
            fine_tune: 是否微调预训练词向量,vectors!=None时有效
            dropout_1: 双向LSTM和单向LSTM之间的dropout比例
            dropout_2: softmax前fc的droupout
            device: 计算设备
        """

        super().__init__(device=device)

        # 预训练 embedding
        if vectors != None:
            self.word_embeds = nn.Embedding(1, 1)
            if fine_tune:
                self.word_embeds = self.word_embeds.from_pretrained(vectors, padding_idx=pad_idx, freeze=False)
            else:
                self.word_embeds = self.word_embeds.from_pretrained(vectors, padding_idx=pad_idx, freeze=True)
        # 随机初始化 embedding
        else:
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # 双向RNN
        self.bi_rnn = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=bi_rnn_num_layers, bidirectional=True,
                              batch_first=True)
        # dropout_1
        self.dropout_1 = nn.Dropout(dropout_1)
        # 单向RNN
        self.uni_rnn = nn.LSTM(hidden_dim // 2 * 2, hidden_dim, num_layers=uni_rnn_num_layers, bidirectional=False,
                               batch_first=True)
        # dropout_2
        self.dropout_2 = nn.Dropout(dropout_2)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, label_size)
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, idx_seqs: torch.tensor):
        """
        根据sentences计算logtis
        Args:
            idx_seqs: 存储词在词典中的索引。[batch_size, sentence_size]

        Returns:softmax的输入

        """

        # idx_seqs = [batch_size, sentence_size]
        embedded = self.word_embeds(idx_seqs)
        # embedded : [batch_size, sentence_size, embedding_dim]

        bi_rnn_out, _ = self.bi_rnn(embedded)
        # bi_rnn_out : [batch_size, sentence_size, hidden_dim // 2 * 2]

        bi_rnn_out = self.dropout_1(bi_rnn_out)
        # bi_rnn_out : [batch_size, sentence_size, hidden_dim // 2 * 2]

        bi_rnn_out = torch.flip(bi_rnn_out, dims=[1])

        uni_rnn_out, _ = self.uni_rnn(bi_rnn_out)
        # uni_rnn_out : [batch_size, sentence_size, hidden_dim]

        uni_rnn_out = uni_rnn_out[:, -1, :]
        # uni_rnn_out : [batch_size, hidden_dim]

        logits = self.fc(self.dropout_2(uni_rnn_out))
        # logits : [batch_size, label_size]

        return logits

    def loss(self, idx_seqs: torch.tensor, idx_labels: torch.tensor):
        """
        TextRNN损失函数
        Args:
            idx_seqs: 存储词在词典中的索引。[batch_size, sentence_size]
            idx_labels: sentences真实labels对应的索引

        Returns:loss

        """
        idx_seqs = idx_seqs.to(self.device)
        # sentences : [batch_size, sentence_size]

        idx_labels = idx_labels.to(self.device)
        # id_labels : [batch_size]

        logits = self(idx_seqs)
        # logits: [batch_size, label_size]

        loss = self.criterion(logits, idx_labels)

        return loss
