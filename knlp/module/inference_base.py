# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference_base
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2022-04-10
# Description:
# -----------------------------------------------------------------------#
from abc import ABC, abstractmethod


class InferenceBaseClass(ABC):
    """
    This function offers the base class to do inference
    Child class could rewrite any function for its specific function.
    """

    def __init__(self):
        pass

    @abstractmethod
    def load_mode(self, ):
        pass

    @abstractmethod
    def predict(self, ):
        pass

    @abstractmethod
    def predict_file(self, ):
        pass
