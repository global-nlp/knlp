#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-28
# Description:
# -----------------------------------------------------------------------#

from abc import ABC
from knlp.seq_labeling.model_train.load_data import DataLoader


class TrainSeqLabel(ABC):
    """
    This function offers the base class to train a new model doing seq_labeling


    """

    def __init__(self, vocab_set_path: str = None, training_data_path: str = None, eval_data_path: str = None,
                 test_data_path: str = None):
        # 训练集
        self.train_data_loader = DataLoader(vocab_set_path=vocab_set_path, training_data_path=training_data_path,
                                            mode="train")
        word2idx, tag2idx = self.train_data_loader.word2idx, self.train_data_loader.tag2idx
        # 验证集
        self.eval_data_loader = DataLoader(eval_data_path=eval_data_path, mode="eval", word2idx=word2idx,
                                           tag2idx=tag2idx) if eval_data_path else None

    def train(self, epoch, print_per=10, eval_per=10):
        """
        训练
        :param epoch:
        :param print_per: 训练多少个batch打印一次训练集损失
        :param eval_per: 训练多少个batch打印一次验证集损失
        :return:
        """
        for _ in range(epoch):
            loss_list = []
            self.model.train()
            for index, (seqs_idx, tags_idx, lengths) in enumerate(self.train_data_loader.datas):
                self.model.zero_grad()
                loss = self.model.loss(seqs_idx, tags_idx, lengths)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

                if len(loss_list) == print_per:
                    print('{0}/{1} train loss:{2}'.format(index * self.train_data_loader.batch_size,
                                                          self.train_data_loader.length, sum(loss_list) / print_per))
                    loss_list = []
                if (index + 1) % eval_per == 0:
                    self.eval()

    def eval(self):
        """
        验证集评估
        :return:
        """
        if self.eval_data_loader:
            self.model.eval()
            for seqs_idx, tags_idx, lengths in self.eval_data_loader.datas:
                loss = self.model.loss(seqs_idx, tags_idx, lengths).item()
                print('eval loss:{0}'.format(loss))

    def test(self):
        pass

    def train_eval_test(self):
        pass

    def _save_nodel(self):
        pass

    @staticmethod
    def load_model(model_path):
        """
        This function could load model.

        Args:
            model_path:

        Returns:

        """
        pass
