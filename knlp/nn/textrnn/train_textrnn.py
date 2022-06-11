# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train_textrnn
# Author: Gong Chen
# Mail: cgg_1996@163.com
# Created Time: 2022-05-04
# Description:
# -----------------------------------------------------------------------#
from knlp.common.constant import PAD
from knlp.nn.textrnn.textrnn import TextRNN
from knlp.nn.textcnn.train_text_classification import TrainTextClassification
from knlp.nn.textcnn.vectors import Vectors
import torch.optim as optim


class TrainTextRNN(TrainTextClassification):
    """
    TextRNN模型训练
    """

    def __init__(self, vectors_path: str = None, model_hyperparameters: dict = {}, optimizer_hyperparameters: dict = {},
                 dataset_hyperparameters: dict = {}, device: str = "cpu"):
        """

        Args:
            vectors_path:
            model_hyperparameters:
            optimizer_hyperparameters:
            dataset_hyperparameters:
            device:
        """
        super().__init__(device=device, **dataset_hyperparameters)
        # 不使用预训练的词向量
        if not vectors_path:
            model_hyperparameters["pad_idx"] = self.word2idx[PAD]
            self.model = TextRNN(self.train_dataset.vocab_size, self.train_dataset.labelset_size,
                                 **model_hyperparameters).to(self.device)

        # 使用预训练的词向量
        elif vectors_path:
            vectors = Vectors(vectors_path, self_stoi=self.word2idx)
            model_hyperparameters["embedding_dim"] = vectors.dim
            model_hyperparameters["pad_idx"] = vectors.pad_index
            model_hyperparameters["vectors"] = vectors.vectors
            self.model = TextRNN(vectors.vocab_size, self.train_dataset.labelset_size,
                                 **model_hyperparameters).to(self.device)
            self.word2idx = vectors.stoi

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
            # "lr": 0.01,
            # "weight_decay": 1e-4
        },
        "model_hyperparameters": {
            "embedding_dim": 64
        },
        "vectors_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_word2vec_100d_22770.txt",

    }
    save_kwargs = {
        "model_path": KNLP_PATH + "/knlp/nn/textrnn/model_textrnn/weibo_model_textrnn.pkl",
        "word2idx_path": KNLP_PATH + "/knlp/nn/textrnn/model_textrnn/weibo_word2idx.json",
        "label2idx_path": KNLP_PATH + "/knlp/nn/textrnn/model_textrnn/weibo_label2idx.json"
    }
    train = TrainTextRNN(**kwargs)
    train.train(20)
    train.save(**save_kwargs)
