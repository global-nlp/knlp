# -*-coding:utf-8-*-
"""
CRF工具函数

word2features:是特征函数模板，这个部分是可自由定义的。
以下的几种特征字典格式都是支持的：
{“string_key”: float_weight, . . . } dict where keys are observed features and values are their weights;
{“string_key”: bool, . . . } dict; True is converted to 1.0 weight, False - to 0.0;
{“string_key”: “string_value”, . . . } dict; that’s the same as {“string_key=string_value”: 1.0, . . . }
[“string_key1”, “string_key2”, . . . ] list; that’s the same as {“string_key1”: 1.0, “string_key2”: 1.0, . . . }
{“string_prefix”: {. . . }} dicts: nested dict is processed and “string_prefix” s prepended to each key
{“string_prefix”: [. . . ]} dicts: nested list is processed and “string_prefix” s prepended to each key
“string_prefix”: set([. . . ])} dicts: nested list is processed and “string_prefix” s prepended to each key
官方API文档：https://sklearn-crfsuite.readthedocs.io/en/latest/

sentence2features:对整个输入序列进行特征提取，循环调用word2features，对每个字进行处理。
"""


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
