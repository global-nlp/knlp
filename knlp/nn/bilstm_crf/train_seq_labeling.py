# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train_seq_labeling
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-03-29
# Description:
# -----------------------------------------------------------------------#

import json
from knlp.nn.bilstm_crf.dataset_seq_labeling import SeqLabelDataset
from knlp.nn.bilstm_crf.train_nn import TrainNN
from torch.utils.data import DataLoader


class TrainSeqLabel(TrainNN):
    """
    This function offers the base class to train a new model doing seq_labeling

    """

    def __init__(self, vocab_set_path: str = None, training_data_path: str = None, eval_data_path: str = None,
                 batch_size: int = 64, shuffle: bool = True, device: str = "cpu"):
        """

        Args:
            vocab_set_path:
            training_data_path:
            eval_data_path:
            batch_size:
            shuffle:
            device:
        """
        super().__init__(device=device)
        # 训练集
        self.train_dataset = SeqLabelDataset(vocab_set_path=vocab_set_path, training_data_path=training_data_path,
                                             mode="train")
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle,
                                            collate_fn=self.train_dataset.collate_fn_train)
        self.word2idx, self.tag2idx = self.train_dataset.word2idx, self.train_dataset.tag2idx
        # 验证集
        if eval_data_path:
            self.eval_dataset = SeqLabelDataset(eval_data_path=eval_data_path, mode="eval", word2idx=self.word2idx,
                                                tag2idx=self.tag2idx)
            self.eval_data_loader = DataLoader(self.eval_dataset, batch_size=self.eval_dataset.__len__(),
                                               collate_fn=self.eval_dataset.collate_fn_train)
        else:
            self.eval_data_loader = None

    def train(self, epoch, print_per=10, eval_per_batch=1000, eval_per_epoch=True):
        """
        训练
        Args:
            epoch:
            print_per: 训练多少个batch打印一次训练集损失
            eval_per: 训练多少个batch打印一次验证集损失

        Returns:

        """
        self.model.train()
        for _ in range(epoch):
            print("#######" * 10)
            print("EPOCH: ", str(_))
            loss_list = []
            for index, (seqs_idx, tags_idx, lengths) in enumerate(self.train_data_loader):
                self.model.zero_grad()
                loss = self.model.loss(seqs_idx, tags_idx, lengths)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

                if len(loss_list) == print_per:
                    print('{0}/{1} train loss:{2}'.format(index * self.train_data_loader.batch_size,
                                                          self.train_dataset.__len__(), sum(loss_list) / print_per))
                    loss_list = []
                if (index + 1) % eval_per_batch == 0:
                    self.eval()
            if eval_per_epoch:
                self.eval()

    def eval(self):
        """
        验证集评估
        Returns:

        """
        if self.eval_data_loader:
            self.model.eval()
            for seqs_idx, tags_idx, lengths in self.eval_data_loader:
                loss = self.model.loss(seqs_idx, tags_idx, lengths).item()
                print('eval loss:{0}'.format(loss))
            self.model.train()

    def test(self):
        pass

    def train_eval_test(self):
        pass

    def save_word2idx(self, word2idx_path):
        """
        保存word2idx
        Args:
            word2idx_path:

        Returns:

        """
        with open(word2idx_path, "w") as f:
            json.dump(self.word2idx, f, indent=4, ensure_ascii=False)

    def save_tag2idx(self, tag2idx_path):
        """
        保存tag2idx
        Args:
            tag2idx_path:

        Returns:

        """
        with open(tag2idx_path, "w") as f:
            json.dump(self.tag2idx, f, indent=4, ensure_ascii=False)
