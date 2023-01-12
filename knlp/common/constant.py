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
delimiter = ' '  # 对于序列标注类型数据统一使用delimiter控制各个代码文件中的分隔符。
sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']
SEED = 0
UNK = "[UNK]"
PAD = "[PAD]"
SEP = "[SEP]"
CLS = "[CLS]"
MASK = "MASK"
model_list = ['hmm', 'crf', 'trie', 'bilstm', 'bert_mrc', 'bert_tagger']    # ner pipeline中目前支持的模型列表
