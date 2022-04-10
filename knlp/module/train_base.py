# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train_base
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2022-04-10
# Description:
# -----------------------------------------------------------------------#

from abc import ABC, abstractmethod


class TrainBaseClass(ABC):
    """
    This function offers the base class to train a new model


    """

    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        """
        General evaluation for the model trained
        Returns:

        """
        pass

    @abstractmethod
    def save_nodel(self):
        pass

    @abstractmethod
    def load_model(self, model_path):
        """
        This function could load model.

        Args:
            model_path:

        Returns:

        """
        pass
