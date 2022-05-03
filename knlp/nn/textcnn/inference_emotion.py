# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference_emotion
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-03-29
# Description:
# -----------------------------------------------------------------------#

from knlp.common.constant import KNLP_PATH
from knlp.nn.textcnn.inference_textcnn import InferenceTextCNN
import jieba


class EmotionInference:
    """
    分词推理
    """

    def __init__(self, model: str = "TextCNN_2_class"):
        """

        Args:
            model: 模型类型
        """
        self.model_infos = {
            "TextCNN_2_class": {
                "kwargs": {
                    "model_path": KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_model_textcnn.pkl",
                    "word2idx_path": KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_word2idx.json",
                    "label2idx_path": KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_label2idx.json",
                    "max_length": 150,
                    "tokenizer": jieba.lcut
                },
                "Inference": InferenceTextCNN
            }
        }
        self.select_inference(model)

    def select_inference(self, model: str):
        """
        Inference配置对应的模型
        Args:
            model: 模型类型

        Returns:

        """
        ModelInference = self.model_infos[model]["Inference"]
        kwargs = self.model_infos[model]["kwargs"]
        self.inference = ModelInference(**kwargs)

    def forward(self, seqs, return_label=True):
        """
        推理运算
        Args:
            seqs:

        Returns:

        """
        if isinstance(seqs, str):
            return self.inference([seqs], return_label=return_label)[0]
        else:
            return self.inference(seqs, return_label=return_label)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == "__main__":
    inference = EmotionInference()
    print(inference("今天发工资了哈哈哈"))
    print(inference("今天发工资了哈哈哈", return_label=False))
    print(inference(["大吉大利，今晚吃鸡"]))
    print(inference(["大吉大利，今晚吃鸡"], return_label=False))
    print(inference(["我好伤心", "上海疫情快要结束了，坚持就是胜利。"]))
    print(inference(["我好伤心", "上海疫情快要结束了，坚持就是胜利。"], return_label=False))
