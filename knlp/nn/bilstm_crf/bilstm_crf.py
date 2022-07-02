# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: bilstm_crf
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-03-24
# Description:
# -----------------------------------------------------------------------#

from knlp.nn.base_nn_model import BaseNNModel
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from TorchCRF import CRF


class BiLSTM_CRF(BaseNNModel):

    def __init__(self, vocab_size: int, tagset_size: int, embedding_dim: int = 64, hidden_dim: int = 64,
                 num_layers: int = 1, seed: int = 2022, device=torch.device("cpu")):
        """
        初始化BiLSTM_CRF模型
        Args:
            vocab_size: 词典的维度
            tagset_size: 标记的维度
            embedding_dim: 词向量的维度
            hidden_dim: 隐藏层的维度
            num_layers: BiLSTM层数
            seed: 随机数种子
            device: 计算设备
        """
        super().__init__(seed=seed, device=device)
        # embedding层,将索引转换为词向量的层。
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # BiLSTM层，输入维度为embedding_dim，一个方向的隐藏层维度为hidden_dim // 2
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True)
        # BiLSTM和CRF之间的全连接层，将LSTM输出的维度hidden_dim映射到tagset_size
        self.hidden2tag = nn.Linear(hidden_dim // 2 * 2, tagset_size)
        # CRF层，用的是TorchCRF
        self.crf = CRF(tagset_size)

    def _get_lstm_out(self, sentences: torch.LongTensor, lengths: list):
        """
        计算lstm的输出
        Args:
            sentences:
            lengths:

        Returns:lstm的输出

        """
        # batch_size > 1时
        if len(sentences) > 1:
            # sentences: [batch_size, sentence_size]

            embeds = self.word_embeds(sentences)
            # embeds: [batch_size, sentence_size, embedding_dim]

            embeds = pack_padded_sequence(embeds, lengths=lengths, batch_first=True)
            # embeds: packed格式
            # embeds[0]: [sum(lengths), embedding_dim]
            # embeds[1]: [sentence_size]  存储lengths信息

            lstm_out, _ = self.lstm(embeds)
            # lstm_out: packed格式
            # lstm_out[0]: [sum(lengths), hidden_dim // 2, direction_num]
            # lstm_out[0]: [sum(lengths), hidden_dim]
            # lstm_out[1]: [sentence_size]  存储lengths信息 lstm_out[1] == embeds[1]

            lstm_out = pad_packed_sequence(lstm_out, batch_first=True)[0]
            # lstm_out: [batch_size, sentence_size, hidden_dim]

        # batch_size == 1时,主要为了推理的加速,减少不必要的环节。
        else:
            # sentences: [1, sentence_size]

            embeds = self.word_embeds(sentences).view(sentences.size()[1], 1, -1)
            # self.word_embeds(sentences): [1, sentence_size, embedding_dim] 其中1为batch_size
            # embeds: [sentence_size, 1, embedding_dim]  其中1为batch_size

            lstm_out, _ = self.lstm(embeds)  # embeds的输入维度为 sentence_size * batch_size * embedding_dim
            # lstm_out: [sentence_size, 1, hidden_dim]  其中1为batch_size

            lstm_out = lstm_out.view(sentences.size()[1], -1).unsqueeze(0)
            # lstm_out: [1, sentence_size, hidden_dim] 其中1为batch_size

        return lstm_out

    def _mask_matrix(self, lengths: list):
        """
        根据lengths生成对应的mask矩阵
        Args:
            lengths: 长度序列

        Returns: mask矩阵 torch.BoolTensor
        例:
        lengths:
        [4,2,2,1]
        mask:
        [[1,1,1,1],
         [1,1,0,0],
         [1,1,0,0],
         [1,0,0,0]]
        """
        # 初始化mask: [batch_size, max_sentence_size]
        # lengths[0]=max(lengths)
        mask = torch.zeros(len(lengths), lengths[0])
        for i, length in enumerate(lengths):
            mask[i][:length] = 1
        return mask.bool().to(self.device)

    def forward(self, sentences: torch.LongTensor, lengths: list):
        """
        根据sentences和lengths 计算得到标注序列
        Args:
            sentences: 存储词在词典中的索引，按长度降序排列。[batch_size, sentence_size]
            lengths: sentences对应的长度值，降序排列。

        Returns:sentences对应的标注序列

        """
        sentences = sentences.to(self.device)
        # sentences: [batch_size, sentence_size]

        lstm_out = self._get_lstm_out(sentences, lengths)
        # lstm_out: [batch_size, sentence_size, hidden_dim]

        emission = self.hidden2tag(lstm_out)
        # lstm_feats: [batch_size, sentence_size, tagset_size]

        mask = self._mask_matrix(lengths)
        # mask: [batch_size, sentence_size]

        tag_seq = self.crf.viterbi_decode(emission, mask)

        return tag_seq

    def loss(self, sentences: torch.LongTensor, tags: torch.LongTensor, lengths: list):
        """
        BiLSTM-CRF损失函数
        Args:
            sentences: 存储词在词典中的索引，按长度降序排列。batch_size * sentence_size
            tags: sentences对应的真实标注序列的索引值。
            lengths: sentences对应的长度值，降序排列。

        Returns:loss

        """
        sentences = sentences.to(self.device)
        # sentences: [batch_size, sentence_size]

        tags = tags.to(self.device)
        # tags: [batch_size, sentence_size]

        lstm_out = self._get_lstm_out(sentences, lengths)
        # lstm_out: [batch_size, sentence_size, hidden_dim]

        emission = self.hidden2tag(lstm_out)
        # lstm_feats: [batch_size, sentence_size, tagset_size]

        mask = self._mask_matrix(lengths)
        # mask: [batch_size, sentence_size]

        loss = -torch.mean(self.crf.forward(emission, tags, mask))

        return loss
