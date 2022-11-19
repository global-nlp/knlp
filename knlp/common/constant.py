# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: constant
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-03-20
# Description:
# -----------------------------------------------------------------------#
import os

KNLP_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../.."
delimiter = ' '  # NER模块pipeline所读入的文件中，每一行内部文本与标签的分隔符
sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']
SEED = 0
UNK = "[UNK]"
PAD = "[PAD]"
SEP = "[SEP]"
CLS = "[CLS]"
MASK = "MASK"
