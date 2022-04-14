# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train_bilstm
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-03-29
# Description:
# -----------------------------------------------------------------------#

from knlp.nn.bilstm_crf.train_seq_labeling import TrainSeqLabel
from knlp.nn.bilstm_crf.bilstm_crf import BiLSTM_CRF
import torch.optim as optim


class TrainBiLSTMCRF(TrainSeqLabel):
    """
    BiLSTMCRF模型训练
    """

    def __init__(self, vocab_set_path: str = None, training_data_path: str = None, eval_data_path: str = None,
                 batch_size: int = 64, model_hyperparameters: dict = {},
                 optimizer_hyperparameters: dict = {"lr": 0.01, "weight_decay": 1e-4}, device: str = "cpu"):
        """

        Args:
            vocab_set_path:
            training_data_path:
            eval_data_path:
            batch_size:
            model_hyperparameters:
            optimizer_hyperparameters:
            device:
        """
        super().__init__(vocab_set_path=vocab_set_path, training_data_path=training_data_path,
                         eval_data_path=eval_data_path, batch_size=batch_size, device=device)
        self.model = BiLSTM_CRF(self.train_dataset.vocab_size, self.train_dataset.tagset_size,
                                **model_hyperparameters).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), **optimizer_hyperparameters)

    def save(self, model_path, word2idx_path, tag2idx_path):
        """
        保存模型文件、配置文件
        Args:
            model_path:
            word2idx_path:
            tag2idx_path:

        Returns:

        """
        self.save_model(model_path)
        self.save_word2idx(word2idx_path)
        self.save_tag2idx(tag2idx_path)


if __name__ == "__main__":
    from knlp.common.constant import KNLP_PATH

    kwargs = {"vocab_set_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_vocab.txt",
              "training_data_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_training_data.txt",
              "batch_size": 64,
              "model_hyperparameters": {
                  "embedding_dim": 64,
                  "hidden_dim": 64,
                  "num_layers": 1
              },
              "optimizer_hyperparameters": {
                  "lr": 0.01,
                  "weight_decay": 1e-4
              }
              }
    save_kwargs = {
        "model_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model/bilstm_crf_seg.pkl",
        "word2idx_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model/word2idx.json",
        "tag2idx_path": KNLP_PATH + "/knlp/nn/bilstm_crf/model/tag2idx.json"
    }
    train = TrainBiLSTMCRF(**kwargs)
    train.train(5)
    train.save(**save_kwargs)
