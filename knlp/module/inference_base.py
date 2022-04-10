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
    def load_model(self, model_path: str):
        """
        This function could load model.

        Args:
            model_path:

        Returns:

        """
        pass

    @abstractmethod
    def predict(self, ):
        """
        use model to do predict

        Returns:

        """
        pass

    @abstractmethod
    def predict_file(self, input_file_path: str, output_file_path: str):
        """
        predict a file.

        Args:
            input_file_path: string
            output_file_path: string

        Returns:

        """

        pass
