# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train_textcnn
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-04-28
# Description:
# -----------------------------------------------------------------------#
from knlp.common.constant import PAD
from knlp.nn.textcnn.textcnn import TextCNN
from knlp.nn.textcnn.train_text_classification import TrainTextClassification
from knlp.nn.textcnn.vectors import Vectors
import torch.optim as optim


class TrainTextCNN(TrainTextClassification):
    """
    TextCNN模型训练
    """

    def __init__(self, static_word2vec_path: str = None, non_static_word2vec_path: str = None,
                 model_hyperparameters: dict = {}, optimizer_hyperparameters: dict = {},
                 dataset_hyperparameters: dict = {}, device: str = "cpu"):
        """

        Args:
            static_word2vec_path:
            non_static_word2vec_path:
            model_hyperparameters:
            optimizer_hyperparameters:
            dataset_hyperparameters:
            device:
        """
        super().__init__(device=device, **dataset_hyperparameters)
        # 不使用预训练的word2vec
        if not static_word2vec_path and not non_static_word2vec_path:
            model_hyperparameters["pad_idx"] = self.word2idx[PAD]
            self.model = TextCNN(self.train_dataset.vocab_size, self.train_dataset.labelset_size,
                                 **model_hyperparameters).to(self.device)
        # static和non_staic同时使用
        if static_word2vec_path and non_static_word2vec_path:
            static_word2vec = Vectors(static_word2vec_path, self_stoi=self.word2idx)
            non_static_word2vec = Vectors(non_static_word2vec_path, self_stoi=self.word2idx)
            model_hyperparameters["embedding_dim"] = static_word2vec.dim + non_static_word2vec.dim
            model_hyperparameters["static_pad_idx"] = static_word2vec.pad_index
            model_hyperparameters["static_vectors"] = static_word2vec.vectors
            model_hyperparameters["non_static_pad_idx"] = non_static_word2vec.pad_index
            model_hyperparameters["non_static_vectors"] = non_static_word2vec.vectors
            self.model = TextCNN(static_word2vec.vocab_size, self.train_dataset.labelset_size,
                                 **model_hyperparameters).to(self.device)
            self.word2idx = static_word2vec.stoi
        # static
        elif static_word2vec_path:
            static_word2vec = Vectors(static_word2vec_path, self_stoi=self.word2idx)
            model_hyperparameters["embedding_dim"] = static_word2vec.dim
            model_hyperparameters["static_pad_idx"] = static_word2vec.pad_index
            model_hyperparameters["static_vectors"] = static_word2vec.vectors
            self.model = TextCNN(static_word2vec.vocab_size, self.train_dataset.labelset_size,
                                 **model_hyperparameters).to(self.device)
            self.word2idx = static_word2vec.stoi
        # non_static
        elif non_static_word2vec_path:
            non_static_word2vec = Vectors(non_static_word2vec_path, self_stoi=self.word2idx)
            model_hyperparameters["embedding_dim"] = non_static_word2vec.dim
            model_hyperparameters["non_static_pad_idx"] = non_static_word2vec.pad_index
            model_hyperparameters["non_static_vectors"] = non_static_word2vec.vectors
            self.model = TextCNN(non_static_word2vec.vocab_size, self.train_dataset.labelset_size,
                                 **model_hyperparameters).to(self.device)
            self.word2idx = non_static_word2vec.stoi

        self.optimizer = optim.Adam(self.model.parameters(), **optimizer_hyperparameters)

    def save(self, model_path, word2idx_path, label2idx_path):
        """
        保存模型文件、配置文件
        Args:
            model_path:
            word2idx_path:
            label2idx_path:

        Returns:

        """
        self.save_model(model_path)
        self.save_word2idx(word2idx_path)
        self.save_label2idx(label2idx_path)


if __name__ == "__main__":
    from knlp.common.constant import KNLP_PATH
    import jieba

    kwargs = {
        "dataset_hyperparameters": {
            "vocab_set_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_vocab_25k.txt",
            # "training_data_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_eval_9988.txt",
            "training_data_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_train_110k.txt",
            # "eval_data_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_eval_9988.txt",
            "tokenizer": jieba.lcut,
            "shuffle": True,
            "batch_size": 64,
            "max_length": 150
        },
        "optimizer_hyperparameters": {
            "lr": 0.01,
            "weight_decay": 1e-4
        },
        "model_hyperparameters": {
            "n_filters": 100,
            "filter_sizes": [3, 4, 5]
        },
        # "non_static_word2vec_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_word2vec_300d_20509.txt",
        "static_word2vec_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_word2vec_100d_22770.txt",

    }
    save_kwargs = {
        "model_path": KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_model_textcnn.pkl",
        "word2idx_path": KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_word2idx.json",
        "label2idx_path": KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_label2idx.json"
    }
    train = TrainTextCNN(**kwargs)
    train.train(5)
    train.save(**save_kwargs)
