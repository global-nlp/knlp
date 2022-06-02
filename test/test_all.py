#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: test
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description:
# -----------------------------------------------------------------------#

from knlp import Knlp
from knlp.utils.util import Trie, get_pytest_data_file, get_model_crf_pinyin_file
from knlp import get_keyword, get_key_sentences, seg, ner, evaluation_seg_files, evaluation_seg, sentiment
from knlp.seq_labeling.trie_seg.inference import TrieInference
from knlp.seq_labeling.pinyin_input_method import inference
from knlp.common.constant import KNLP_PATH
from knlp.seq_labeling.crf.crf import CRFModel

import time
from knlp.utils import util

TEST_SINGLE_SENTENCE = "KNLP是一个NLP工具包，主要支持中文的各种NLP基础操作"


def test_knlp():
    with open(get_pytest_data_file(), encoding='utf-8') as f:
        text = f.read()
    res = Knlp(text)
    print("seg_result is", res.seg_result)
    print("seg_result_hmm is", res.seg_result_hmm)
    print("ner_result is", res.ner_result)
    print("sentiment score is", res.sentiment)
    print("key_words are", res.key_words)
    print("key sentences are", res.key_sentences)
    gold_string = '就读 于 中国人民大学 电视 上 的 电影 节目 项目 的 研究 角色 本人 将 会 参与 配音'
    pred_string = '就读 于 中国 人民 大学 电视 上 的 电影 节目 项 目的 研究 角色 本人 将 会 参与 配音'
    print("evaluation res are", res.evaluation_segment(gold_string, pred_string))
    abs_path_to_gold_file = 'test/data/gold_data.txt'
    abs_path_to_pred_file = 'test/data/pred_data.txt'
    gold_file_name = f'{abs_path_to_gold_file}'
    pred_file_name = f'{abs_path_to_pred_file}'
    eval_res = res.evaluation_segment_file(gold_file_name, pred_file_name)
    print(f"evaluation file res are: precision {eval_res[0]}, recall {eval_res[1]}, f1score {eval_res[2]}")


def test_seg():
    res = seg(TEST_SINGLE_SENTENCE)
    print(res)


def test_ner():
    res = ner(TEST_SINGLE_SENTENCE)
    print(res)


def test_get_keyword():
    res = get_keyword(TEST_SINGLE_SENTENCE)
    print(res)


def test_get_key_sentences():
    res = get_key_sentences(TEST_SINGLE_SENTENCE)
    print(res)


def test_sentiment():
    res = sentiment(TEST_SINGLE_SENTENCE)
    print(res)


def test_single_sentence_evaluation():
    gold_string = '就读 于 中国人民大学 电视 上 的 电影 节目 项目 的 研究 角色 本人 将 会 参与 配音'
    pred_string = '就读 于 中国 人民 大学 电视 上 的 电影 节目 项 目的 研究 角色 本人 将 会 参与 配音'
    res = evaluation_seg(gold_string, pred_string, seg_symbol=" ")
    print(res)


def test_file_evaluation():
    abs_path_to_gold_file = 'test/data/gold_data.txt'
    abs_path_to_pred_file = 'test/data/pred_data.txt'
    gold_file_name = f'{abs_path_to_gold_file}'
    pred_file_name = f'{abs_path_to_pred_file}'
    res = evaluation_seg_files(gold_file_name, pred_file_name, seg_symbol=" ")
    print(res)


def test_check_file():
    start = time.time()
    util.check_file("../knlp/data")
    print(time.time() - start)


def test_Trie():
    """
        trie树获插入、获取前缀、获取词频测试
    Returns:

    """
    test_trie = Trie()
    test_trie.insert("北", 20)
    test_trie.insert("南", 20)
    test_trie.insert("北京", 10)
    test_trie.insert("北京大学", 50)
    print(test_trie.trie)
    print(test_trie.find_all_prefix("北京大学"))
    print(test_trie.get_words_freq("北京"))
    print(test_trie.get_words_freq("北大"))


def test_cut_by_knlp():
    trieTest = TrieInference()
    print(trieTest.knlp_seg("测试分词的结果是否符合预期"))


def test_pinyin_inference():
    """
           对于拼音输入法inference的全部进行测试

       """
    test = inference.Inference()
    CRFModel()
    to_be_pred = "dongtianlailechuntianyejiangdaolai"
    test.spilt_predict(to_be_pred, get_model_crf_pinyin_file())
    print("POS结果：" + str(test.label_prediction))
    print("拼音分割结果：" + str(test.out_sentence))
    observe = test.out_sentence
    out = []
    for idx in range(0, len(observe), 2):
        if idx + 1 < len(observe):
            res = test.viterbi(observations=observe[idx:idx + 2])
            print(res)
            out.extend(res[0][1])
        else:
            res = test.viterbi(observations=observe[idx:idx + 1])
            print(res)
            out.extend(res[0][1])
    print("按照两个字一组划分后的预测结果：" + str(out))


def test_all():
    test_knlp()
    test_seg()
    test_ner()
    test_get_keyword()
    test_get_key_sentences()
    test_single_sentence_evaluation()
    test_file_evaluation()
    test_Trie()
    test_cut_by_knlp()
    test_pinyin_inference()
    test_check_file()   # 文件check，暂时先不上线


if __name__ == '__main__':
    test_all()
