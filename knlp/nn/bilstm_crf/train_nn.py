# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train_nn
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-03-29
# Description:
# -----------------------------------------------------------------------#


import torch


class TrainNN:
    """
    This function offers the base class to train a new model doing seq_labeling


    """

    def __init__(self, device: str = "cpu"):
        """

        Args:
            device:
        """
        self._set_device(device)

    def train(self, epoch, print_per=10, eval_per=10):
        """
        训练
        Args:
            epoch:
            print_per: 训练多少个batch打印一次训练集损失
            eval_per: 训练多少个batch打印一次验证集损失

        Returns:

        """
        pass

    def eval(self):
        """
        验证集评估
        Returns:

        """
        pass

    def test(self):
        pass

    def train_eval_test(self):
        pass

    def save_model(self, model_path: str):
        """
        保存模型
        Args:
            model_path:

        Returns:

        """
        torch.save(self.model, model_path)

    def load_model(self, model_path: str):
        """
        This function could load model.

        Args:
            model_path:

        Returns:

        """
        self.model = torch.load(model_path)

    def _set_device(self, device: str):
        """
        设置设备
        :param device:
        :return:
        """
        self.device = torch.device("cuda") if device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
