# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: util
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-03-17
# Description: 完成包括输入数据的归一化（可能针对不同的模块需求不同），一些常用的小函数等
# 主要包括以下的小型模块：
# 1. 多线程类 （最后想了想，好像没用啊）
# 2. 函数运行进度条
# 3. 计算函数运行时间的装饰器
# -----------------------------------------------------------------------#
import os
import sys
import time
import shutil
import zipfile

import requests

from functools import wraps

from knlp.common.constant import KNLP_PATH


# from knlp.seq_labeling.NER.trie_seg.inference import TrieInference


def get_jieba_dict_file():
    return KNLP_PATH + "/knlp/data/NER_data/jieba_dict.txt"


def get_wait_to_cut_file():
    return KNLP_PATH + "/knlp/data/NER_data/wait_to_cut.txt"


def get_default_stop_words_file():
    return KNLP_PATH + "/knlp/data/NER_data/stopwords.txt"


def get_ner_dict_file():
    return KNLP_PATH + "/knlp/data/NER_data/ner_dict.txt"


class Trie:
    """
        字典树：将词库构建成字典树，以便快速搜索前缀
        trie：字典类型、表示整个树
        freq_total：所有词的词频之和

        应用：创建一个Trie对象，使用insert插入逐条插入字词即可构建字典树
    """

    def __init__(self):
        self.trie = {}
        self.freq_total = 0

    def insert(self, dict_words, freq, feature):
        """
            向trie中插入词语，将词语的每一个进行插入。如果字存在，则判断词的下一个字是否存在，如果不存在则建立一颗子树
        Args:
            dict_words: 词库中的词
            freq:  词频

        Returns:

        """
        current_node = self.trie
        for char in dict_words:
            if char not in current_node:
                current_node[char] = {}
            current_node = current_node[char]
        current_node['freq'] = freq  # 词语结束记录词频，同时作为标记
        current_node['feature'] = feature
        # print(self.trie)

    def del_word(self, dict_words):
        current_node = self.trie
        for char in dict_words:
            if char not in current_node:
                print(f"该词: '{dict_words}' 本就不存在！")
                return
            current_node = current_node[char]
        current_node.clear()
        # print(self.trie)

    def find_all_prefix(self, words):
        """
            对于输入词，获取该词在词库中存在的所有前缀 （"北京大学" 所有前缀："北"、"北京"、"北京大学"）
        Args:
            words: 待获取前缀的词语

        Returns:

        """
        current_node = self.trie
        result = set()
        for i in range(len(words)):
            # 判断当前节点下是否存在words[i]对应的字
            if words[i] not in current_node:
                break
            else:
                current_node = current_node[words[i]]
                if 'freq' in current_node:
                    result.add((words[0:i + 1], current_node['freq']))

        return result if len(result) != 0 else None

    def get_words_freq(self, words):
        """
            获取指定词语words的词频
        Args:
            words:

        Returns:

        """
        current_node = self.trie
        for i in range(len(words)):
            if words[i] in current_node:
                current_node = current_node[words[i]]
            else:
                return None

        return current_node['freq'] if 'freq' in current_node else None

    def get_words_feature(self, words):
        """
            获取指定词语words的属性
        Args:
            words:

        Returns:

        """
        current_node = self.trie
        for i in range(len(words)):
            if words[i] in current_node:
                current_node = current_node[words[i]]
            else:
                return None

        return current_node['feature'] if 'feature' in current_node else None

    def get_from_dict(self, words, feature):
        """
            获取指定词语words的属性
        Args:
            words:

        Returns:

        """
        current_node = self.trie
        for i in range(len(words)):
            if words[i] in current_node:
                current_node = current_node[words[i]]
            else:
                return None
        if 'from_dict' in current_node:
            if current_node['feature'] != feature:
                return None
            else:
                return current_node['from_dict']
        else:
            return None


class PostProcessTrie(Trie):
    def __init__(self):
        super().__init__()
        self.text_dict = {}
        self.tag_list = []
        self.entity_set = set()
        self.user_dict_index_with_feature = set()
        self.convert_feature_dict = {
            "com": "company",
            "boo": "book",
            "gam": "game",
            "nam": "name",
            "mov": "movie",
            "org": "organization",
            "pos": "position",
            "add": "address",
            "sce": "scene",
            "gov": "government",
            None: "O"
        }

    def get_entity(self):
        return self.entity_set

    def insert(self, dict_words, feature, _from):
        """
            向trie中插入词语，将词语的每一个进行插入。如果字存在，则判断词的下一个字是否存在，如果不存在则建立一颗子树
        Args:
            dict_words: 词库中的词

        Returns:

        """
        current_node = self.trie
        for char in dict_words:
            if char not in current_node:
                current_node[char] = {}
            current_node = current_node[char]
        current_node['feature'] = feature
        if _from == 'dict':
            current_node['from_dict'] = True
        if _from == 'pred':
            current_node['from_dict'] = False
        # print(self.trie)

    def find_all_prefix(self, words):
        """
            对于输入词，获取该词在词库中存在的所有前缀 （"北京大学" 所有前缀："北"、"北京"、"北京大学"）
        Args:
            words: 待获取前缀的词语

        Returns:

        """
        current_node = self.trie
        result = set()
        for i in range(len(words)):
            # 判断当前节点下是否存在words[i]对应的字
            if words[i] not in current_node:
                break
            else:
                current_node = current_node[words[i]]
                if 'feature' in current_node:
                    result.add((words[0:i + 1], current_node['feature'], current_node['from_dict']))

        return result if len(result) != 0 else None

    def construct_text_dict(self, text, tag_list):
        self.trie.clear()
        self.text_dict.clear()
        self.tag_list.clear()
        self.entity_set.clear()
        self.user_dict_index_with_feature.clear()
        flag = 0
        begin = 0
        for index, tag in enumerate(tag_list):
            if tag[0] != 'O' and flag == 0:
                begin = index
                flag = 1
            elif (tag[0] != 'I' and flag == 1) or (tag[0] == 'I' and index == len(tag_list) - 1):
                flag = 0
                # print(begin, index)
                if index == len(tag_list) - 1:
                    index += 1
                piece = ''.join(text[begin:index])
                if piece not in self.text_dict.keys():
                    self.text_dict[piece] = {
                        'feature': tag_list[begin][2:]
                    }
                if tag[0] != 'O':
                    begin = index
                    flag = 1
                # print(self.text_dict)
            elif tag != 'O' and flag == 1:
                index += 1

        self.construct_trie()
        # print(self.freq_total)

    def construct_trie(self):
        for contain in self.text_dict.keys():
            # print(contain)
            self.insert(contain, self.text_dict[contain]['feature'], _from='pred')
        # print(self.trie)

    def post_soft_process(self, sentence, entity_set):
        """

        :param entity_set:
        :param sentence: 后处理句子
        :return:
        """
        self.entity_set = entity_set
        DAG = self.get_DAG(sentence)
        for position in DAG.keys():
            if DAG[position][0] == position:
                feature = self.get_words_feature(sentence[position])
                if self.get_from_dict(sentence[position], feature):
                    self.entity_set.add((sentence[position], feature))
            else:
                for end_pos in DAG[position]:
                    words = sentence[position: end_pos + 1]
                    feature = self.get_words_feature(words)
                    if self.get_from_dict(words, feature):
                        self.entity_set.add((words, feature))

    def post_hard_process(self, sentence, entity_set):
        """

        :param entity_set:
        :param sentence: 后处理句子
        :return:
        """
        self.entity_set = entity_set
        DAG = self.get_DAG(sentence)
        for position in DAG.keys():
            if DAG[position][0] == position:
                feature = self.get_words_feature(sentence[position])
                if self.get_from_dict(sentence[position], feature):
                    self.entity_set.add((sentence[position], feature))
            else:
                for end_pos in DAG[position]:
                    words = sentence[position: end_pos + 1]
                    feature = self.get_words_feature(words)
                    if self.get_from_dict(words, feature):
                        self.entity_set.add((words, feature))
                    elif str(position)+feature in self.user_dict_index_with_feature:
                        self.entity_set.remove((words, feature))

    def convert_label_list(self, sentence, feature):
        for idx, tag in enumerate(feature):
            if tag == 'O':
                self.tag_list.append(tag)
            else:
                entity = sentence[idx]
                for index, char in enumerate(entity):
                    if index == 0:
                        self.tag_list.append('B' + '-' + tag)
                    else:
                        self.tag_list.append('I' + '-' + tag)

    def get_DAG(self, sentence):
        """
        遍历句子的每一个字，获取sentence[idx:-1]所有前缀，构成有向无环图
        Args:
            sentence: 待分词的句子或文本

        Returns: 得到的有向无环图

        """
        DAG = {}
        for i in range(len(sentence)):
            arr = []
            all_prefix_words = self.find_all_prefix(sentence[i:])

            if all_prefix_words is None:
                # sentence[i:] 在词库中获取不到前缀时，i位置的路径就是i
                arr.append(i)
            else:
                # 把每一个前缀词的结束位置添加到数组 例：DAG[200] = [200,202,204]  说明200这个位置有三条路径可选
                for words in all_prefix_words:
                    arr.append(len(words[0]) - 1 + i)  # word[0] 前缀词，word[1] 词频，word[2] 是否来自用户字典
                    if words[2]:
                        self.user_dict_index_with_feature.add(str(i)+words[1])
            DAG[i] = arr
        return DAG


def check_file(file_path):
    """
    检测数据文件是否存在，不存在则进行下载。
    目前用于测试，将knlp/data数据文件上传到 https://github.com/Kevin1906721262/knlp-file ，
    利用github的zip下载形式，下载到本地，并解压到对应位置。
    暂未实现多个模块下的数据文件检测(也可实现,统一下载，解压后移动到不同模块下即可)

    存在的问题：国内有时候连不上github，会经常出现连接不上的情况。
    数据文件80多M, github项目10M左右，该方式下载网络好的时候就几秒，慢的时候几十秒

    Args:
        file_path: string, 待检测文件夹路径

    Returns:

    """
    if not os.path.exists(file_path):  # "../knlp/data"
        origin_file_url = "https://github.com/Kevin1906721262/knlp-file" \
                          "/archive/refs/heads/main.zip "
        if not os.path.exists("../tmp"):
            os.mkdir("../tmp")
        temp_file_path = "../tmp/main.zip"
        try:
            f = requests.get(origin_file_url)
        except Exception as e:
            print(e)
            print("网络异常，数据文件下载失败")
        else:
            with open(temp_file_path, "wb") as code:
                code.write(f.content)
            z = zipfile.ZipFile(temp_file_path, 'r')
            z.extractall(path="../tmp/")
            z.close()

            shutil.move("../tmp/knlp-file-main/data", "../knlp/data")
            shutil.rmtree("../tmp")


class AttrDict(dict):
    """Dict that can get attribute by dot"""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def funtion_time_cost(function):
    """
    打印出原始function的耗时
    Args:
        function: 被装饰器装饰的方程

    Python装饰器（decorator）在实现的时候，被装饰后的函数其实已经是另外一个函数了
    （函数名等函数属性会发生改变）
    Python的functools包中提供了一个叫wraps的decorator来消除这样的副作用。
    写一个decorator的时候，在实现之前加上functools的wrap
    它能保留原有函数的名称和docstring。就像下面的这个实现。

    Returns: wrapper

    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        print(f"function {function.__name__} begin running")
        time_start = time.time()
        res = function(*args, **kwargs)
        time_end = time.time()
        print(f"time cost is {time_end - time_start} and running over")
        return res

    return wrapper


class ShowProcess:
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0  # 当前的处理进度
    max_steps = 0  # 总共需要处理的次数
    max_arrow = 50  # 进度条的长度

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)  # 计算显示多少个'>'
        num_line = self.max_arrow - num_arrow  # 计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps  # 计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']' \
                      + '%.2f' % percent + '%' + '\r'  # 带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar)  # 这两句打印字符到终端
        sys.stdout.flush()

    def close(self, words='done'):
        print('')
        print(words)
        self.i = 0
