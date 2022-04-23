# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: seg_app
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2022-04-06
# Description:
# -----------------------------------------------------------------------#

from knlp.seq_labeling import seg, seg_hmm, seg_crf


def jieba_seg_word():
    word = "做事情不能自嗨，要真的能用才行"
    res = seg(word)
    print(res)


def hmm_seg_word():
    word = "做事情不能自嗨，要真的能用才行"
    res = seg_hmm(word)
    print(res)


def crf_seg_word():
    word = "做事情不能自嗨，要真的能用才行"
    res = seg_crf(word)
    print(res)


if __name__ == '__main__':
    jieba_seg_word()
    hmm_seg_word()
    crf_seg_word()
