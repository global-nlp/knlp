# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: __init__.py
# Author: FengQin Liu
# Mail: 1906721262@qq.com
# Created Time: 2022-03-27
# Description: 用于trie分词的实现
# -----------------------------------------------------------------------#

from knlp.utils.util import get_jieba_dict_file, Trie
from math import log


class TrieInference:

    def __init__(self, dict_file=get_jieba_dict_file()):
        """
        初始化字典树
        :param dict_file:
        """
        self._trie = Trie()
        with open(dict_file, 'r', encoding='utf-8') as f:
            for word in f:
                self._trie.insert(word.split(" ")[0], word.split(" ")[1])
                self._trie.freq_total += int(word.split(" ")[1])

    def knlp_seg(self, sentence):
        DAG = get_DAG(sentence, self._trie)
        route = get_route(DAG, sentence, self._trie)

        # 根据记录的路径返回结果
        i = 0
        result = []
        while i < len(sentence):
            stop = route[i][1] + 1
            result.append(sentence[i:stop])
            i = stop
        return result


def get_DAG(sentence, trie):
    """
    遍历获取sentence[idx:-1]所有前缀，构成有向无环图
    :param sentence: 待分词的句子
    :param trie: 构建好的字典树
    :return:
    """
    DAG = {}
    for i in range(len(sentence)):
        arr = []
        words = trie.find_all_prefix(sentence[i:])

        if not words:
            # TODO 一个前缀都没有的情况,暂不处理
            pass
        for word in words:
            arr.append(len(word[0]) - 1 + i)  # word[0] 前缀词，word[1] 词频
        DAG[i] = arr
    return DAG


def get_route(DAG, sentence, trie):
    """
    求大概率路径思路: 从后往前遍历,求出sentence[idx:-1]的最大概率路径及概率
    P(idx) = max(P(sentence[idx:x]) + P(sentence[x:-1])) , x in DAG[idx]
    :param DAG:
    :param sentence:
    :param trie:
    :return:
    """
    N = len(sentence)
    route = {N: (0, 0)}  # route 存储idx位置 最大概率及对应路径
    log_freq_total = log(trie.freq_total)  # 使用对数计算防止溢出
    for idx in range(N - 1, -1, -1):
        temp_list = []  # 临时存放idx位置，各个前缀的词频及路径
        for x in DAG[idx]:
            words_freq = trie.get_words_freq(sentence[idx:x + 1])
            # [idx:-1] 的概率 由两部分组成，后一部分已计算过
            idx_freq = log(int(words_freq) or 1) - log_freq_total + route[x + 1][0]
            temp_list.append((idx_freq, x))
        route[idx] = max(temp_list)
    return route


if __name__ == '__main__':
    test_trie = Trie()
    test_trie.insert("北", 20)
    test_trie.insert("南", 20)
    test_trie.insert("北京", 10)
    test_trie.insert("北京大学", 50)
    print(test_trie.trie)
    print(test_trie.find_all_prefix("北京大学"))
    print(test_trie.get_words_freq("北大"))
