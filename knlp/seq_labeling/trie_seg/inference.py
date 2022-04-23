# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: __init__.py
# Author: FengQin Liu
# Mail: 1906721262@qq.com
# Created Time: 2022-03-27
# Description: 用于trie分词的实现
# -----------------------------------------------------------------------#
from math import log

from knlp.utils.util import get_jieba_dict_file, Trie


class TrieInference:

    def __init__(self, dict_file=get_jieba_dict_file()):
        """
            初始化字典树
        Args:
            dict_file: 词库文件位置
        """
        self._trie = Trie()
        with open(dict_file, 'r', encoding='utf-8') as f:
            # knlp/data/jieba_dict.txt 词库文件,获取的word为文件一行。每一行三个元素分别为(词，词频，词性) 其中 词性对照 n-名词 z-状态词 nz-状态词 v-动词 m-数量词
            # r-代词 t-时间词 等等。词性类型较多 详见：jieba分词词性对照表 https://blog.csdn.net/u013317445/article/details/117925312
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
    遍历句子的每一个字，获取sentence[idx:-1]所有前缀，构成有向无环图
    Args:
        sentence: 待分词的句子或文本
        trie: 构建好的字典树

    Returns: 得到的有向无环图

    """
    DAG = {}
    for i in range(len(sentence)):
        arr = []
        all_prefix_words = trie.find_all_prefix(sentence[i:])

        if all_prefix_words is None:
            # sentence[i:] 在词库中获取不到前缀时，i位置的路径就是i
            arr.append(i)
        else:
            # 把每一个前缀词的结束位置添加到数组 例：DAG[200] = [200,202,204]  说明200这个位置有三条路径可选
            for words in all_prefix_words:
                arr.append(len(words[0]) - 1 + i)  # word[0] 前缀词，word[1] 词频
        DAG[i] = arr
    return DAG


def get_route(DAG, sentence, trie):
    """
    求大概率路径思路: 从后往前遍历,求出sentence[idx:-1]的最大概率路径及概率
    P(idx) = max(P(sentence[idx:x]) + P(sentence[x:-1])) , x in DAG[idx]
    Args:
        DAG: 待分句子获取文本构成的有向无环图
        sentence: 待分句子或文本
        trie: 构建好的字典树

    Returns: 计算得到的最大概率路径

    """
    N = len(sentence)
    route = {N: (0, 0)}  # route 存储idx位置 最大概率及对应路径
    log_freq_total = log(trie.freq_total)  # 使用对数计算防止溢出
    for idx in range(N - 1, -1, -1):
        temp_list = []  # 临时存放idx位置，各个前缀的词频及路径
        for x in DAG[idx]:
            words_freq = trie.get_words_freq(sentence[idx:x + 1])
            # [idx:-1] 的概率 由两部分组成，后一部分已计算过
            freq = 1 if words_freq is None else int(words_freq)
            idx_freq = log(freq) - log_freq_total + route[x + 1][0]
            temp_list.append((idx_freq, x))
        route[idx] = max(temp_list)
    return route


if __name__ == '__main__':
    trieTest = TrieInference()
    print(get_DAG("测试分词的结果是否符合预期", trieTest._trie))
    print(trieTest.knlp_seg("测试分词的结果是否符合预期"))
