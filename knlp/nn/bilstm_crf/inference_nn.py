# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference_nn
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-04-08
# Description:
# -----------------------------------------------------------------------#

import torch


class InferenceNN:
    """
    该类为神经网络推理父类
    """

    def __init__(self, model_path: str = "", model: torch.nn.Module = None, device: str = "cpu"):
        """

        Args:
            model_path: 模型路径
            device:
        """
        # 传入模型 或 传入模型路径
        assert model or model_path
        self._set_device(device)
        if model:
            self.model = model
        else:
            self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str):
        """
        加载模型
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

    @staticmethod
    def restore_sort(datas: list, restore_indexs: list):
        return [datas[index] for index in restore_indexs]
