# -*-coding:utf-8-*-
import pickle

from codecs import open
from sklearn_crfsuite import CRF
from knlp.common.constant import KNLP_PATH

class CRFModel(object):

    def __init__(self):
        self.model = CRF(algorithm='lbfgs',
                         c1=0.1,
                         c2=0.1,
                         max_iterations=100,
                         all_possible_transitions=False)

    def train(self, sentences, tag_lists):
        features = [sent2features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    def test(self, sentences):
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)

    return model


def crf_train(train_data):
    # 训练CRF模型
    train_words, train_tags = train_data
    crf_model = CRFModel()
    crf_model.train(train_words, train_tags)
    save_model(crf_model, KNLP_PATH+"/knlp/model/crf/crf.pkl")


# ******** CRF 工具函数*************


def word2features(sent, i):
    """抽取单个字的特征"""
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i - 1]
    next_word = "</s>" if i == (len(sent) - 1) else sent[i + 1]
    # 使用的特征：
    # 前一个词，当前词，后一个词，
    # 前一个词+当前词， 当前词+后一个词
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word + word,
        'w:w+1': word + next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    """抽取序列特征"""
    return [word2features(sent, i) for i in range(len(sent))]
