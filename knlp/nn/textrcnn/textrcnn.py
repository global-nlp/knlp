# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: textrcnn
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-05-04
# Description:
# -----------------------------------------------------------------------#

from knlp.nn.base_nn_model import BaseNNModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(BaseNNModel):

    def __init__(self, vocab_size: int, label_size: int, pad_idx: int, embedding_dim: int = 64, hidden_dim: int = 64,
                 num_layers: int = 1, vectors: torch.tensor = None, fine_tune=False, dropout: float = 0.5,
                 device=torch.device("cpu")):
        """
        初始化TextRNN模型
        Args:
            vocab_size: 词典的维度
            label_size: 类别的维度
            pad_idx: embedding中PAD的索引
            embedding_dim: 词向量的维度
            hidden_dim: 隐藏层的维度
            num_layers: 双向LSTM的层数
            vectors: 预训练词向量
            fine_tune: 是否微调预训练词向量,vectors!=None时有效
            dropout: droupout
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
        # 双向LSTM
        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True,
                               batch_first=True)
        # dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc_1 = nn.Linear(hidden_dim // 2 * 2 + embedding_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, label_size)
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

        bi_lstm_out, _ = self.bi_lstm(embedded)
        # bi_rnn_out : [batch_size, sentence_size, hidden_dim // 2 * 2]

        embedded = torch.cat([embedded, bi_lstm_out], 2)
        # embedded : [batch_size, sentence_size, embedding_dim + hidden_dim // 2 * 2]

        fc_1_result = self.fc_1(embedded)
        # fc_1_result : [batch_size, sentence_size, hidden_dim]

        fc_1_result = fc_1_result.permute(0, 2, 1)
        # fc_1_result : [batch_size, hidden_dim, sentence_size]

        pooled = F.max_pool1d(fc_1_result, fc_1_result.shape[2]).squeeze(2)
        # pooled : [batch_size, hidden_dim]

        pooled = self.dropout(torch.tanh(pooled))
        # pooled : [batch_size, hidden_dim]

        logits = self.fc_2(pooled)
        # logits : [batch_size, label_size]

        return logits

    def loss(self, idx_seqs: torch.tensor, idx_labels: torch.tensor):
        """
        TextRCNN损失函数
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
