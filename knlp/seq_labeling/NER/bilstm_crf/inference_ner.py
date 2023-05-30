# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference_seg
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-04-08
# Description:
# -----------------------------------------------------------------------#
import re

from knlp.common.constant import KNLP_PATH
from knlp.nn.bilstm_crf.inference_bilstm_crf import InferenceBiLSTMCRF
from knlp.seq_labeling.NER.Inference.Inference import NERInference


class BilstmInference(NERInference):
    """
    分词推理
    """

    def __init__(self, model: str = "BiLSTM_CRF"):
        """

        Args:
            model: 模型类型
        """
        super().__init__()
        self.model_infos = {
            "BiLSTM_CRF": {
                "kwargs": {
                    "model_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/bilstm_crf_ner_msra.pkl",
                    "word2idx_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/word2idx.json",
                    "tag2idx_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/tag_json.json"
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

    def forward(self, seqs):
        """
        推理运算
        Args:
            seqs:

        Returns:

        """
        if not self.is_zh:
            seqs = list(seqs.split())
        self.cut_bio(seqs, self.inference([seqs])[0])
        return self.get_sent()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_tag(self):
        return self.tag_list

    def get_entity(self):
        return self.entity_set


if __name__ == "__main__":
    inference = BilstmInference()
    print(inference("中南大学计算机学院，今天天气不错"))
    print(inference.get_tag())
    print(inference.get_entity())
