# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: textcnn
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-04-28
# Description:
# -----------------------------------------------------------------------#

from knlp.nn.base_nn_model import BaseNNModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(BaseNNModel):
    def __init__(self, vocab_size: int, label_size: int, n_filters: int, filter_sizes: list, embedding_dim: int = 64,
                 pad_idx: int = None, static_pad_idx: int = None, non_static_pad_idx: int = None,
                 static_vectors: torch.tensor = None, non_static_vectors: torch.tensor = None, dropout: float = 0.5,
                 seed: int = 2022, device=torch.device("cpu")):
        """
        初始化TextCNN模型
        Args:
            vocab_size: 词典的维度
            label_size: 类别的维度
            n_filters: 卷积输出的通道数
            filter_sizes: 卷积核尺寸list
            embedding_dim: 词向量的维度
            pad_idx: embedding中PAD的索引，没有预训练词向量时必填
            static_pad_idx: static预训练词向量中PAD的索引
            non_static_pad_idx: non_static预训练词向量中PAD的索引
            static_vectors: static预训练词向量
            non_static_vectors: non_static预训练词向量
            dropout: softmax前全连接的dropout比例
            seed: 随机数种子
            device: 计算设备
        """
        super().__init__(seed=seed, device=device)
        assert (static_vectors != None and static_pad_idx) or \
               (non_static_vectors != None and non_static_pad_idx) or \
               (pad_idx and embedding_dim)
        # embedding层,将索引转换为词向量的层。
        self.embedding_list = []
        # 预训练 static embedding
        if static_vectors != None and static_pad_idx:
            self.static_word_embeds = nn.Embedding(1, 1)
            self.static_word_embeds = self.static_word_embeds.from_pretrained(static_vectors,
                                                                              padding_idx=static_pad_idx,
                                                                              freeze=True)
            self.embedding_list.append(self.static_word_embeds)
        # 预训练 non_static embedding
        if non_static_vectors != None and non_static_pad_idx:
            self.non_static_word_embeds = nn.Embedding(1, 1)
            self.non_static_word_embeds = self.non_static_word_embeds.from_pretrained(non_static_vectors,
                                                                                      padding_idx=non_static_pad_idx,
                                                                                      freeze=False)
            self.embedding_list.append(self.non_static_word_embeds)
        # 随机初始化 embedding
        if len(self.embedding_list) == 0 and pad_idx:
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.embedding_list.append(self.word_embeds)

        # 确保non_static 和 static使用相同大小的词典。
        if len(self.embedding_list) == 2:
            assert self.embedding_list[0].state_dict()["weight"].shape[0] == \
                   self.embedding_list[1].state_dict()["weight"].shape[0]
        # 自适应embedding_dim
        embedding_dim = sum([embedding.embedding_dim for embedding in self.embedding_list])

        # 卷积层集合
        self.conv2d_list = []
        for index, filter_size in enumerate(filter_sizes):
            conv2d_name = "conv2d_{}".format(index)
            conv2d = nn.Conv2d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=(filter_size, 1))
            self.add_module(conv2d_name, conv2d)
            self.conv2d_list.append(conv2d)

        # dropout
        self.dropout = nn.Dropout(dropout)
        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * n_filters, label_size)
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
        embedded_list = []
        for embedding in self.embedding_list:
            embedded = embedding(idx_seqs)
            # embedded : [batch_size, sentence_size, embedding_dim]
            embedded = embedded.permute(0, 2, 1)
            # embedded : [batch_size, embedding_dim, sentence_size]
            embedded = embedded.unsqueeze(-1)
            # embedded : [batch_size, embedding_dim, sentence_size, 1]
            embedded_list.append(embedded)

        embedded = torch.cat(embedded_list, dim=1)
        # embedded : [batch_size, in_channels, sentence_size, 1]

        pooled_list = []
        for conv2d in self.conv2d_list:
            conved = F.relu(conv2d(embedded).squeeze(3))
            # conv_n(embedded) : [batch_size, n_filters, sentence_size - filter_sizes[n] + 1, 1]
            # conved : [batch_size, n_filters, sentence_size - filter_sizes[n] + 1]

            pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
            # F.max_pool1d(conved_n, conved_n.shape[2]) : [batch_size, n_filters, 1]
            #                                           : [batch_size, out_channels, 1]
            # pooled : [batch_size, n_filters]
            #        : [batch_size, out_channels]
            pooled_list.append(pooled)

        cat = self.dropout(torch.cat(pooled_list, dim=1))
        # cat : [batch_size, n_filters * len(filter_sizes)]

        logits = self.fc(cat)
        # logits : [batch_size, label_size]

        return logits

    def loss(self, idx_seqs: torch.tensor, idx_labels: torch.tensor):
        """
        TextCNN损失函数
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
