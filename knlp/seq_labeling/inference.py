#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description:
# -----------------------------------------------------------------------#

import time
from threading import Thread

from knlp.seq_labeling.ner import NER
from knlp.seq_labeling.seg import Segmentor
from knlp.utils.util import get_wait_to_cut_file


def seg(sentence, function_name="jieba_cut"):
    """
        This function could call different function to cut sentence

    Args:
        sentence: string
        function_name: string

    Returns: list of word

    """

    if "knlp_cut".__eq__(function_name):
        return Segmentor.trie_seg(sentence=sentence, model="test")

    words = []
    seg = Segmentor()
    word_list = seg.segment(text=sentence, function_name=function_name)

    for word in word_list:
        word = word.strip()
        if not word:
            continue
        words.append(word)

    return words


def ner(sentence, function_name="jieba_ner"):
    """
    This function could return the ner res of sentence via different function

    Args:
        sentence: string
        function_name: string

    Returns: list of pairs (word, tag)

    """
    word_tags = []
    ner_method = getattr(NER, function_name, None)
    if not ner_method:
        # TODO raise an exception
        return None
    for word_tag in ner_method(sentence):
        if not word_tag:
            continue
        word_tags.append(word_tag)
    return word_tags


def knlp_cut(text):
    """
        knlp 分词测试
    """
    start_time = time.time()
    result_knlp = seg(txt_data, "knlp_cut")
    print("knlp分词耗时：", int(round(time.time() * 1000)) - int(round(start_time * 1000)))
    with open("../data/result_knlp.txt", "w", encoding="utf-8") as knlp_file:
        knlp_file.write(str(result_knlp))


def jieba_cut(text):
    """
        jieba 分词测试
    """
    start_time = time.time()
    result_jieba = seg(txt_data)
    print("jieba分词耗时：", int(round(time.time() * 1000)) - int(round(start_time * 1000)))
    with open("../data/result_jieba.txt", "w", encoding="utf-8") as knlp_file:
        knlp_file.write(str(result_jieba))


if __name__ == '__main__':
    wait_file = get_wait_to_cut_file()
    with open(wait_file, "r", encoding="utf-8") as f:
        txt_data = f.read()

    jieba_cut(txt_data)
    knlp_cut(txt_data)

    # 两个线程 同时处理
    t1 = Thread(target=knlp_cut, args=(txt_data,))
    t2 = Thread(target=jieba_cut, args=(txt_data,))
    t2.start()
    t1.start()
