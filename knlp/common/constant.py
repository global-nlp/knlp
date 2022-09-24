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

GIT_DATA_URL = "https://github.com/global-nlp/knlp_data/archive/refs/heads/main.zip"  # /knlp/data 数据下载位置
GIT_MODEL_URL = "https://github.com/global-nlp/knlp_model/archive/refs/heads/main.zip"  # /knlp/model 数据下载位置
KNLP_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../.."
sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']
SEED = 0
UNK = "[UNK]"
PAD = "[PAD]"
SEP = "[SEP]"
CLS = "[CLS]"
MASK = "MASK"
