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
import datetime
import os
import re
import shutil
import sys
import time
import zipfile
from functools import wraps

import requests
from knlp.common.constant import KNLP_PATH


def get_jieba_dict_file():
    return KNLP_PATH + "/knlp/data/jieba_dict.txt"


def get_wait_to_cut_file():
    return KNLP_PATH + "/knlp/data/wait_to_cut.txt"


def get_default_stop_words_file():
    return KNLP_PATH + "/knlp/data/stopwords.txt"


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


class CommonUtil(object):
    _emoji_pattern_cfg = re.compile(u'('
                                    u'\ud83c[\udf00-\udfff]|'
                                    u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                                    u'[\u2600-\u2B55])+', flags=re.UNICODE)

    _replace_pattern_cfg = {
        'float_t': re.compile('\d+\.\d+'),
        'phone_t': re.compile(
            r'1[0-9\*]{10}|\d{3}[-\s]\d{4}[-\s]\d{4}|\+861[0-9]{10}|[0-9]{3}-[0-9]{3}-[0-9]{4}|[0-9]{4}-[0-9]{7,8}|[8|6][0-9]{7}'),
        'email_t': re.compile(r'[^@|\s]+@[^@]+\.[^@|\s]+'),
    }

    replace_patterns = [
        ('', re.compile(r'\[.*\]'))
    ]

    punc = '!"#$%&\'()*＊+。,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~}'
    replace_symbol = (r'[{}]+'.format(punc))
    _replace_symbol_cfg = re.compile(replace_symbol)

    _illegal_char_set = set([])

    @classmethod
    def keep_alphabate(cls, line):
        cop = re.compile("[^a-z^A-Z]")
        return cop.sub('', line)

    @classmethod
    def keep_meaning_thing(cls, line):
        cop = re.compile(
            "[^\u4e00-\u9fa5^.^a-z^A-Z^0-9^.^\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b”]")
        return cop.sub('', line)

    @classmethod
    def keep_Chinese(cls, line):
        cop = re.compile("[^\u4e00-\u9fa5”]")
        return cop.sub('', line)

    @classmethod
    def keep_Chinese_number(cls, line):
        cop = re.compile("[^\u4e00-\u9fa5^.^0-9”]")
        return cop.sub('', line)

    @classmethod
    def rm_replace_symbol(cls, line):
        res = cls._replace_symbol_cfg.sub('', line)
        return res

    @classmethod
    def remove_emoji_char(cls, text_unicode):
        res = cls._emoji_pattern_cfg.sub('', text_unicode)
        return res

    @classmethod
    def rm_html(cls, line):
        pat = re.compile('>(.*?)<')
        return ''.join(pat.findall(line))

    # 时间函数相关
    @classmethod
    def timestamp2ymd(cls, timestamp, format='%Y-%m-%d %H:%M:%S'):
        # timestamp: 1494474258
        # format:
        #     - '%Y-%m-%d %H:%M:%S'
        #     - '%Y-%m-%d'
        #     - '%Y%m%d'
        x = time.localtime(timestamp)
        return time.strftime(format, x)

    @classmethod
    def ymd2timestamp(cls, Ymd, format='%Y-%m-%d %H:%M:%S'):

        # Ymd: 2017-04-13 00:00:00
        # format:
        # 	- '%Y-%m-%d %H:%M:%S'
        # 	- '%Y-%m-%d'
        # 	- '%Y%m%d'
        # return 1494474258

        return time.mktime(time.strptime(Ymd, format))

    @classmethod
    def get_timestamp(cls, now_flag=1, days=0):
        # now_flag: 是否获取当前时间戳，精确到秒
        # days: 某天开始时间戳
        if now_flag == 1:
            return time.time()
        else:
            today = datetime.date.today()
            # 今天-num天时间戳
            TimeStamp = today + datetime.timedelta(days=days)
            return TimeStamp

    def _get_charyype(cls, character):
        # Input:
        #   unicode of a character
        # Return:
        #   0-number, 1-alpha, 2-Chinese, 3-others
        char = character
        if len(char) != 1:
            return -1
        if char in '1234567890':
            return 0  # number
        if (char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z'):
            return 1
        if re.match(u'[\u4E00-\u9FA5]', char):
            return 2
        return 3

    @classmethod
    def get_char_typenum(cls, text):
        # Input:
        #   unicode of text
        # Return:
        #   [#number, #alpha, #Chinese, #Others]
        fNumOfNum, fNumOfAlpha, fNumOfChn, fNumOfOther = 0, 0, 0, 0
        for char in text:
            charType = cls._get_charyype(char)
            if 0 == charType:
                fNumOfNum += 1
            elif 1 == charType:
                fNumOfAlpha += 1
            elif 2 == charType:
                fNumOfChn += 1
            elif 3 == charType:
                fNumOfOther += 1
        return fNumOfNum, fNumOfAlpha, fNumOfChn, fNumOfOther

    @classmethod
    def is_chinese(cls, uchar):
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            return True
        else:
            return False

    @classmethod
    def is_alphabet(cls, uchar):
        if (uchar >= u'\u0041' and uchar <= u'\u005a') or \
                (uchar >= u'\u0061' and uchar <= u'\u007a'):
            return True
        else:
            return False

    @classmethod
    def is_number(cls, uchar):
        if uchar >= u'\u0030' and uchar <= u'\u0039':
            return True
        else:
            return False

    @classmethod
    def q2b(cls, uchar):
        """全角转半角"""
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
            return uchar
        return unichr(inside_code)

    @classmethod
    def str_q2b(cls, sequence):
        """把字符串全角转半角"""
        sequence = sequence.decode('utf-8')
        return "".join([cls.q2b(uchar) for uchar in sequence]).encode('utf-8').strip()

    @classmethod
    def str_q2b_u2l(cls, sequence):
        """格式化字符串，完成全角转半角，大写转小写的工作"""
        return cls.str_q2b(sequence).lower().strip()
