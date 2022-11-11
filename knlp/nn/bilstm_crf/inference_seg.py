# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference_seg
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-04-08
# Description:
# -----------------------------------------------------------------------#

from knlp.common.constant import KNLP_PATH
# from knlp.nn.bilstm_crf.inference_bilstm_crf import InferenceBiLSTMCRF
from knlp.nn.bilstm_crf.inference_bilstm_crf import InferenceBiLSTMCRF


class SegInference:
    """
    分词推理
    """

    def __init__(self, model: str = "BiLSTM_CRF"):
        """

        Args:
            model: 模型类型
        """
        self.model_infos = {
            "BiLSTM_CRF": {
                "kwargs": {
                    "model_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/bilstm_crf_seg.pkl",
                    "word2idx_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/word2idx.json",
                    "tag2idx_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/tag2idx.json"
                },
                "Inference": InferenceBiLSTMCRF
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

    def cut(self, sentence1, sentence2):
        """
        按照BEMS标签做中文分词，切割句子。
        Args:
            sentence1: 文本序列
            sentence2: 标注序列

        Returns:

        """
        out_sent = []
        begin = 0
        for idx in range(len(sentence1)):
            if sentence2[idx] == 'B':
                begin = idx
            elif sentence2[idx] == 'S':
                str = sentence1[idx]
                out_sent.append(str)
                begin = idx + 1
            elif sentence2[idx] == 'E':
                next = idx + 1
                str = "".join(sentence1[begin:next])
                out_sent.append(str)
                begin = next
        return out_sent

    def forward(self, seqs):
        """
        推理运算
        Args:
            seqs:

        Returns:

        """
        if isinstance(seqs, str):
            return self.cut(seqs, self.inference([seqs])[0])
        else:
            tags_idx = self.inference(seqs)
            return [self.cut(seq, idx) for seq, idx in zip(seqs, tags_idx)]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == "__main__":
    inference = SegInference()
    print(inference("你好，吃晚饭了吗"))
    print(inference(["冬天到了，春天还会远吗？", "天安门前太阳升"]))
    print(inference(["冬天到了，春天还会远吗？", "今天晚上我们一起去吃大餐好不好？", "你好，吃晚饭了吗"]))
