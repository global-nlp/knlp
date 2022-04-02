# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: base_nn_model
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-03-29
# Description:
# -----------------------------------------------------------------------#

from knlp.seq_labeling.model_train.train_base import TrainSeqLabel
from knlp.nn.bilstm_crf.bilstm_crf import BiLSTM_CRF
import torch.optim as optim


class TrainBiLSTMCRF(TrainSeqLabel):

    def __init__(self, vocab_set_path=None, training_data_path=None, eval_data_path=None):
        super().__init__(vocab_set_path=vocab_set_path, training_data_path=training_data_path,
                         eval_data_path=eval_data_path)
        self.model = BiLSTM_CRF(self.train_data_loader.vocab_size, self.train_data_loader.tagset_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)


if __name__ == "__main__":
    from knlp.common.constant import KNLP_PATH

    args = {"vocab_set_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_vocab.txt",
            "training_data_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_training_data.txt",
            "eval_data_path": KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_test_data.txt"}

    train = TrainBiLSTMCRF(**args)
    train.train(5)
