# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train_text_classification
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-03-29
# Description:
# -----------------------------------------------------------------------#

import json
from knlp.nn.textcnn.dataset_text_classification import TextClassificationDataset
from knlp.nn.bilstm_crf.train_nn import TrainNN
from torch.utils.data import DataLoader


class TrainTextClassification(TrainNN):
    """
    This function offers the base class to train a new model doing text classification

    """

    def __init__(self, vocab_set_path: str = None, training_data_path: str = None, eval_data_path: str = None,
                 batch_size: int = 64, shuffle: bool = True, tokenizer=None, max_length: int = 100,
                 device: str = "cpu"):
        """

        Args:
            vocab_set_path:
            training_data_path:
            eval_data_path:
            batch_size:
            shuffle:
            tokenizer:
            device:
        """
        super().__init__(device=device)
        # 训练集
        self.train_dataset = TextClassificationDataset(vocab_set_path=vocab_set_path,
                                                       training_data_path=training_data_path, tokenizer=tokenizer,
                                                       max_length=max_length, mode="train")
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.word2idx, self.label2idx = self.train_dataset.word2idx, self.train_dataset.label2idx
        # 验证集
        if eval_data_path:
            self.eval_dataset = TextClassificationDataset(eval_data_path=eval_data_path, mode="eval",
                                                          word2idx=self.word2idx, label2idx=self.label2idx,
                                                          tokenizer=tokenizer, max_length=max_length)
            self.eval_data_loader = DataLoader(self.eval_dataset, batch_size=self.eval_dataset.__len__())
        else:
            self.eval_data_loader = None

    def train(self, epoch, print_per=10, eval_per=1000, eval_per_epoch=True):
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
            loss_list = []
            for index, (seqs_idx, labels_idx) in enumerate(self.train_data_loader):
                self.model.zero_grad()
                loss = self.model.loss(seqs_idx, labels_idx)
                loss.backward()

                self.optimizer.step()

                loss_list.append(loss.item())

                if len(loss_list) == print_per:
                    print('{0}/{1} train loss:{2}'.format(index * self.train_data_loader.batch_size,
                                                          self.train_dataset.__len__(), sum(loss_list) / print_per))
                    loss_list = []
                if (index + 1) % eval_per == 0:
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
            for seqs_idx, labels_idx in self.eval_data_loader:
                loss = self.model.loss(seqs_idx, labels_idx).item()
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

    def save_label2idx(self, label2idx_path):
        """
        保存label2idx
        Args:
            label2idx_path:

        Returns:

        """
        with open(label2idx_path, "w") as f:
            json.dump(self.label2idx, f, indent=4, ensure_ascii=False)
