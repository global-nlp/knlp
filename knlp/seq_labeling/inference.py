#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description:
# -----------------------------------------------------------------------#

from knlp.seq_labeling.ner import NER
from knlp.seq_labeling.seg import Segmentor
import time
from knlp.utils.util import Trie
from knlp.utils.util import get_stop_words_train_file


def knlp_seg(sentence, function_name="knlp_cut"):
    """

    Args:
        sentence: string
        function_name: string

    Returns: list of word

    """
    seg = Segmentor(stop_words_file=get_stop_words_train_file())
    word_list = seg.trie_seg(sentence=sentence, model="test")
    return word_list


def seg(sentence, function_name="jieba_cut"):
    """
        This function could call different function to cut sentence

    Args:
        sentence: string
        function_name: string

    Returns: list of word

    """

    if "knlp_cut".__eq__(function_name):
        word_list = Segmentor.trie_seg(sentence=sentence, model="test")
        print("knlp分词结果", end="：")
        return word_list

    words = []
    seg = Segmentor(stop_words_file=get_stop_words_train_file())
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





def trie_test():
    trie = Trie()
    trie.insert("北", 20)
    trie.insert("南", 20)
    trie.insert("北京", 10)
    trie.insert("北京大学", 50)
    # print(trie.trie)
    # print(trie.find_all_trie("北京大学123"))
    # print(trie.find_all_trie("南"))
    # print(trie.find_all_trie("太棒了"))
    print(trie.get_words_freq("北大"))


if __name__ == '__main__':
    start_time = time.time()
    # print(seg("测试分词的结果是否符合预期"))
    print(seg("测试分词的结果是否符合预期", "knlp_cut"))
    print(int(round(time.time() * 1000)) - int(round(start_time * 1000)))

    # trie_test()




