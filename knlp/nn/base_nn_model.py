# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: base_nn_model
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-09-04
# Description:
# -----------------------------------------------------------------------#
import os
import random
import torch
import torch.nn as nn
from knlp.common.constant import SEED


class BaseNNModel(nn.Module):
    """
    This class made a base model, all nn related method will be derived from this class

    """

    def __init__(self, device: str = torch.device("cpu")):
        super(BaseNNModel, self).__init__()
        self.device = device
        self.set_seed(SEED)

    def set_seed(self, seed: int):
        """
        设置随机数种子
        Args:
            seed:

        Returns:

        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
