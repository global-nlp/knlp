#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference
# Author: FengQing Liu
# Mail: liu_F_Q@163.com
# Created Time: 2022-05-06
# Description:
# -----------------------------------------------------------------------#

import sys
import time
import jieba

from knlp import seg
from threading import Thread
from knlp.seq_labeling.trie_seg.inference import TrieInference
from knlp.utils.util import get_jieba_dict_file


def knlp_cut(text):
    """
        knlp 分词测试
    """
    start_time = time.time()
    result_knlp = seg(text, "trie_seg")
    print("knlp分词耗时：", int(round(time.time() * 1000)) - int(round(start_time * 1000)))
    with open("../../data/result_knlp.txt", "w", encoding="utf-8") as knlp_file:
        knlp_file.write(str(result_knlp))
    return result_knlp


def jieba_cut(text):
    """
        jieba 分词测试
    """
    start_time = time.time()
    result_jieba = seg(text)
    print("jieba分词耗时：", int(round(time.time() * 1000)) - int(round(start_time * 1000)))
    with open(get_jieba_dict_file(), "w", encoding="utf-8") as knlp_file:
        knlp_file.write(str(result_jieba))


def compare_knlp_jieba_cut(txt_data):
    jieba_cut(txt_data)
    knlp_cut(txt_data)

    # 两个线程 同时处理
    t1 = Thread(target=knlp_cut, args=(txt_data,))
    t2 = Thread(target=jieba_cut, args=(txt_data,))
    t2.start()
    t1.start()


def compare_knlp_jieba_init():
    """
    从字典树构建上评估一下两种结果的效果
    Returns:

    """
    print("knlp分词字典树构建测试：")
    start = time.time()
    L = TrieInference()
    print(round(time.time() - start, 2), "s")
    memory_size = round(sys.getsizeof(L._trie.trie) / 1024 / 1024, 2)
    print(memory_size, "M")

    print("jieba分词字典树构建测试：")
    start = time.time()
    f = open(get_jieba_dict_file(), 'rb')
    L, S = jieba.Tokenizer.gen_pfdict(f)
    print(round(time.time() - start, 2), "s")
    memory_size = round(sys.getsizeof(L) / 1024 / 1024, 2)
    print(memory_size, "M")


if __name__ == '__main__':
    # 通过句子测试分词
    # txt_data = "测试分词的结果是否符合预期"
    # print(knlp_cut(txt_data))

    # 通过文本测试分词
    # with open(get_wait_to_cut_file(), "r", encoding="utf-8") as f:
    #     txt_data = f.read()
    # knlp_cut(txt_data)
    # jieba_cut(txt_data)

    # 比较两种分词结果和耗时
    # compare_knlp_jieba_cut(txt_data)

    # 比较初始化空间和时间占用
    compare_knlp_jieba_init()
