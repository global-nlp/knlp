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
from knlp.utils import util
from knlp.common.constant import DATA_DIR, MODEL_DIR


def get_model_crf_hanzi_file():
    file_name = KNLP_PATH + "/knlp/model/crf/hanzi_segment.pkl"
    util.check_file(file_name, MODEL_DIR)
    return file_name


def get_model_crf_pinyin_file():
    file_name = KNLP_PATH + "/knlp/model/crf/pinyin.pkl"
    util.check_file(file_name, MODEL_DIR)
    return file_name


def get_pku_hmm_train_file():
    file_name = KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_training_data_sample.txt"
    util.check_file(file_name, DATA_DIR)
    return file_name


def get_pku_vocab_train_file():
    file_name = KNLP_PATH + "/knlp/data/seg_data/train/pku_vocab.txt"
    util.check_file(file_name, DATA_DIR)
    return file_name


def get_pytest_data_file():
    file_name = KNLP_PATH + "/knlp/data/pytest_data.txt"
    util.check_file(file_name, DATA_DIR)
    return file_name


def get_jieba_dict_file():
    file_name = KNLP_PATH + "/knlp/data/jieba_dict.txt"
    util.check_file(file_name, DATA_DIR)
    return file_name


def get_wait_to_cut_file():
    file_name = KNLP_PATH + "/knlp/data/wait_to_cut.txt"
    util.check_file(file_name, DATA_DIR)
    return file_name


def get_default_stop_words_file():
    file_name = KNLP_PATH + "/knlp/data/stopwords.txt"
    util.check_file(file_name, DATA_DIR)
    return file_name


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

    def insert(self, dict_words, freq):
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


def check_file(file_path, dir_type):
    """
    检测数据文件是否存在，不存在则进行下载。
    目前用于测试，将knlp/data数据文件上传到 https://github.com/global-nlp/knlp_data ，
    model : https://github.com/global-nlp/knlp_model
    利用github的zip下载形式，下载到本地，并解压到对应位置。
    暂未实现多个模块下的数据文件检测(也可实现,统一下载，解压后移动到不同模块下即可)

    存在的问题：国内有时候连不上github，会经常出现连接不上的情况。
    数据文件80多M, github项目10M左右，该方式下载网络好的时候就几秒，慢的时候几十秒

    Args:
        file_path: string, 待检测文件夹路径
        dir_type: string, 标识是需要下载knlp/data 还是下载 knlp/model
    Returns:

    """
    if not os.path.exists(file_path):
        origin_file_url = "https://github.com/liuFQ47/knlp_" + dir_type + "/archive/refs/heads/main.zip"
        os.chdir(KNLP_PATH)
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        temp_file_path = "tmp/main.zip"
        try:
            print(file_path, "is not exist, init data file, downing...")
            f = requests.get(origin_file_url)
            print("data file down finished")
        except Exception as e:
            print(e)
            raise e
        else:
            with open(temp_file_path, "wb") as code:
                code.write(f.content)
            z = zipfile.ZipFile(temp_file_path, 'r')
            z.extractall(path="tmp/")
            z.close()
            if os.path.exists("knlp/" + dir_type):
                shutil.rmtree("knlp/" + dir_type)
            shutil.move("tmp/knlp_" + dir_type + "-main/" + dir_type, "knlp/")
            shutil.rmtree("tmp")


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
