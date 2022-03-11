# -*-coding:utf-8-*-
# ******** CRF 工具函数*************

def word2features(sent, i):
    """特征函数模板，抽取单个字的特征"""
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


def sentence2features(sentence):
    """抽取序列特征"""
    return [word2features(sentence, i) for i in range(len(sentence))]
