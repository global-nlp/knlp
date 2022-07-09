# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference_bilstm_crf
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-04-08
# Description:
# -----------------------------------------------------------------------#

from knlp.nn.bilstm_crf.inference_seq_labeling import InferenceSeqLabel
import torch


class InferenceBiLSTMCRF(InferenceSeqLabel):
    """
    用于BiLSTMCRF推理
    """

    def __init__(self, model_path: str = "", model: torch.nn.Module = None, word2idx_path: str = "",
                 tag2idx_path: str = "", device: str = "cpu"):
        """

        Args:
            model_path:
            model:
            word2idx_path:
            tag2idx_path:
            device:
        """
        super().__init__(model_path=model_path, model=model, word2idx_path=word2idx_path,
                         tag2idx_path=tag2idx_path, device=device)


if __name__ == "__main__":
    from knlp.common.constant import KNLP_PATH

    kwargs = {
        "model_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/bilstm_crf_seg.pkl",
        "word2idx_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/word2idx.json",
        "tag2idx_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/tag2idx.json"
    }
    inference = InferenceBiLSTMCRF(**kwargs)

    print(inference(["冬天到了，春天还会远吗？", "天安门前太阳升"]))
    print(inference(["冬天到了，春天还会远吗？", "今天晚上我们一起去吃大餐好不好？", "你好，吃晚饭了吗"]))
