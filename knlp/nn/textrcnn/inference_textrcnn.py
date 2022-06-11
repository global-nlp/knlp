# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference_textrnn
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-05-04
# Description:
# -----------------------------------------------------------------------#

from knlp.nn.textcnn.inference_text_classification import InferenceTextClassification
import torch


class InferenceTextRCNN(InferenceTextClassification):
    """
    用于TextRCNN推理
    """

    def __init__(self, model_path: str = "", model: torch.nn.Module = None, word2idx_path: str = "",
                 label2idx_path: str = "", max_length: int = None, tokenizer=None, preprocess_fn=None,
                 device: str = "cpu"):
        """

        Args:
            model_path:
            model:
            word2idx_path:
            label2idx_path:
            max_length:
            tokenizer:
            preprocess_fn:
            device:
        """
        super().__init__(model_path=model_path, model=model, word2idx_path=word2idx_path,
                         label2idx_path=label2idx_path, max_length=max_length, tokenizer=tokenizer,
                         preprocess_fn=preprocess_fn, device=device)


if __name__ == "__main__":
    from knlp.common.constant import KNLP_PATH
    import jieba

    kwargs = {
        "model_path": KNLP_PATH + "/knlp/nn/textrcnn/model_textrcnn/weibo_model_textrcnn.pkl",
        "word2idx_path": KNLP_PATH + "/knlp/nn/textrcnn/model_textrcnn/weibo_word2idx.json",
        "label2idx_path": KNLP_PATH + "/knlp/nn/textrcnn/model_textrcnn/weibo_label2idx.json",
        "max_length": 150,
        "tokenizer": jieba.lcut
    }
    inference = InferenceTextRCNN(**kwargs)

    print(inference(["大吉大利，今晚吃鸡"]))
    print(inference(["大吉大利，今晚吃鸡"], return_label=True))
    print(inference(["[怒]", "郁闷"]))
    print(inference(["开心", "好[害怕]呀"]))
    print(inference(["[担忧]", "[怎么办]"]))