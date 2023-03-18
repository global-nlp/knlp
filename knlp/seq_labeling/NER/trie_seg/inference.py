# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: __init__.py
# Author: Ziyang Miao
# Mail: 1838040569@qq.com
# Created Time: 2022-05-25
# Description: 用于trie命名实体识别的实现
# -----------------------------------------------------------------------#
from math import log

from knlp.seq_labeling.NER.Inference.Inference import NERInference
from knlp.seq_labeling.NER.trie_seg.ner_util import get_ner_dict_file, Trie


class TrieInference(NERInference):

    def __init__(self, dict_file=get_ner_dict_file()):
        """
            初始化字典树
        Args:
            dict_file: 词库文件位置
        """
        super().__init__()
        self._trie = Trie()
        with open(dict_file, 'r', encoding='utf-8') as f:
            # knlp/data/jieba_dict.txt 词库文件,获取的word为文件一行。每一行三个元素分别为(词，词频，词性) 其中 词性对照 n-名词 z-状态词 nz-状态词 v-动词 m-数量词
            # r-代词 t-时间词 等等。词性类型较多 详见：jieba分词词性对照表 https://blog.csdn.net/u013317445/article/details/117925312
            for word in f:
                self._trie.insert(word.split(" ")[0], word.split(" ")[1], word.split(" ")[-1].split("\n")[0])
                self._trie.freq_total += int(word.split(" ")[1])

    def add_word(self, word_freq):
        self._trie.insert(word_freq[0], word_freq[1], word_freq[2])
        self._trie.freq_total += int(word_freq[1])

    def del_word(self, word_freq):
        self._trie.del_word(word_freq[0])

    def knlp_seg(self, sentence):
        DAG = self.get_DAG(sentence, self._trie)
        route = self.get_route(DAG, sentence, self._trie)

        """
            route: 示例(计算得到的每一个字 最大概率的停顿位置)
            {
                7: (0,0),
                6: (-9.503728473394347,6),
                5: (-8.367386874813025,6),
                4: (-13.606542360673144,4),
                3: (-22.863752458864894,3),
                2: (-28.573656509261923,3),
                1: (-38.26292456314442,1),
                0: (-38.683818316725656,1)
            }
            其中 对于 0: (-38.683818316725656,1)  0 表示第一个字，后面元组第一个元素为最大概率路径的概率值，
                元组第二个元素为最大概率的停顿位置。
                因此结果为01/23/4/56
            
        """
        # 将最优路径连成句
        i = 0
        result = []
        while i < len(sentence):
            stop = route[i][1] + 1
            result.append(sentence[i:stop])
            i = stop
        features = self.get_feature(result)
        self.convert_label_list(result, features)
        return result, features

    def get_DAG(self, sentence, trie):
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

    def get_route(self, DAG, sentence, trie):
        """
            例 “北京大学的图书馆”
            D[idx]：第idx个字 所有前缀词长度数组（也就是可能的停顿位置）
                DAG[1] = (1,2,4)  (假设)第一个位置“北” 在词库中对应的有三个前缀词 北，北京，北京大学

            R[idx] 第idx个字的最大概率以及停顿位置
                如R[1]= (-38.683818316725656,2) 这说明第1个字在2停顿的概率是最大的，最大概率是-38.683818316725656

            递推公式dp:
                R[idx] = max(F(sentence[idx:x]) + R[x][0] , x) x in DAG[idx]
                F(sentence[idx:x]) 表示sentence[idx:x]词频 (也就是 北，北京，北京大学 的词频)

                简单说 第1个字，最大概率的停顿位置是通过比较 P(北)+P(北:-1) 、 P(北京)+P(大:-1) 、 P(北京大学)+P(的:-1)
                三条路径的概率 来确定的，谁的概率大就停在哪里，将结果以元组的形式记录在R[1]中
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
                freq = 1 if words_freq is None else int(words_freq)  # 如果未获取到词频就置1, log(1) = 0
                idx_freq = log(freq) - log_freq_total + route[x + 1][0]
                temp_list.append((idx_freq, x))
            route[idx] = max(temp_list)
        return route

    def get_feature(self, result):
        list_out = []
        for words in result:
            feature = self._trie.get_words_feature(words)
            if not feature:
                feature = 'O'
            list_out.append(feature)
        return list_out

    def convert_label_list(self, sentence, feature):
        self.entity_set.clear()
        for idx, tag in enumerate(feature):
            if tag == 'O':
                self.tag_list.append(tag)
            else:
                entity = sentence[idx]
                self.entity_set.add((entity, tag))
                for index, char in enumerate(entity):
                    if index == 0:
                        self.tag_list.append('B' + '-' + tag)
                    else:
                        self.tag_list.append('I' + '-' + tag)


if __name__ == '__main__':
    print('Trie构建开始')
    trieTest = TrieInference()
    print('Trie构建结束')

    print(trieTest.get_DAG("你会和星级厨师一道先从巴塞罗那市中心兰布拉大道的laboqueria市场的开始挑选食材，", trieTest._trie))
    str = ('英国馆', '30', 'company')
    trieTest.add_word(str)
    print(trieTest.knlp_seg("也就是说英国人在世博会上的英国馆，不会相办法表现出我是英国馆，从我的传统角度、文化角度，"))
    print(trieTest.get_entity())

    trieTest.del_word(str)
    print(trieTest.knlp_seg("也就是说英国人在世博会上的英国馆，不会相办法表现出我是英国馆，从我的传统角度、文化角度，"))
    print(trieTest.get_tag())
    print(trieTest.get_entity())
