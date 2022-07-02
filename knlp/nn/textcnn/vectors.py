# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: vectors
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-04-28
# Description:
# -----------------------------------------------------------------------#

from knlp.common.constant import UNK, PAD
import torchtext.vocab
import torch


class Vectors(torchtext.vocab.Vectors):
    """
    在torchtext.vocab.Vectors的基础上进行开发，读取训练好的word2vec
    在其基础上加入了:
    (1)把UNK,PAD添加到itos和stoi中，并可获得索引。
    (2)可以根据给定的stoi/itos重新构建vectors，避免词典过大。
    """

    def __init__(self, name: str, cache: str = None, url: str = None, unk_init=None, max_vectors: int = None,
                 self_itos: list = None, self_stoi: dict = None):
        """

        Args:
            name: word2vec文件路径
            cache: 缓存路径
            url: 下载地址
            unk_init: unk初始化方式
            max_vectors: 最大词典大小
            self_itos: 自定义词典
            self_stoi: 自定义word2idx
        """
        super().__init__(name, cache=cache, url=url, unk_init=unk_init, max_vectors=max_vectors)
        # 不提供self_itos/self_stoi，使用预训练词典
        if not self_itos and not self_stoi:
            self.itos = [UNK, PAD] + self.itos
            self.stoi = dict({UNK: 0, PAD: 1}, **{word: index + 2 for word, index in self.stoi.items()})
            self.vectors = torch.cat([self.unk_init(torch.Tensor(2, self.dim)), self.vectors], dim=0)
        # 使用self_itos的词典
        elif self_itos:
            if UNK in self_itos:
                self_itos.pop(self_itos.index(UNK))
            if PAD in self_itos:
                self_itos.pop(self_itos.index(PAD))
            itos = [UNK, PAD] + self_itos
            stoi = dict(zip(itos, list(range(len(itos)))))
            vectors = torch.cat([self.get_vecs_by_tokens(token).unsqueeze(0) for token in itos], dim=0)
            self.itos = itos
            self.stoi = stoi
            self.vectors = vectors
        # 使用self_stoi的词典
        elif self_stoi:
            if UNK in self_stoi:
                self_stoi.pop(UNK)
            if PAD in self_stoi:
                self_stoi.pop(PAD)
            itos = [UNK, PAD] + list(self_stoi.keys())
            stoi = dict(zip(itos, list(range(len(itos)))))
            vectors = torch.cat([self.get_vecs_by_tokens(token).unsqueeze(0) for token in itos], dim=0)
            self.itos = itos
            self.stoi = stoi
            self.vectors = vectors
        self.unk_index = 0
        self.pad_index = 1
        self.vocab_size = len(self.itos)
