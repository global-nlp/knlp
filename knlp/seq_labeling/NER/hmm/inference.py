# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-03-18
# Description:
# -----------------------------------------------------------------------#
import json
import re

from knlp.seq_labeling.NER.Inference.Inference import Inference
from knlp.utils.util import label_list
from knlp.common.constant import KNLP_PATH


class HMMInference(Inference):
    """
    hmm 的本质便是利用之前基于统计数据计算出来的几个概率，针对输入的sequence进行正向计算，以得到想要的结果
    应该具备的功能点
    1 load 模型和


    首先需要梳理，拿到一个sentence之后，要怎么计算获取到最终的分词结果。
    然后再考虑使用viterbi进行
    最后再想想怎么进行新词发现

    最后整体上过一遍工程，把相关的内容整理成文案，发到知乎，并发布在各个群里面，还有公众号里面。

    """

    def __init__(self, training_data_path):
        """
        _end_set:实体标签中的最后一个标签
        """
        super().__init__()
        # clue:
        self._end_set = ("O", "I-movie", "I-organization", "I-address", "I-scene", "I-name", "I-position", "I-government", "I-game",
            "I-book", "I-company")
        # coll:
        # self._end_set = ("O", "I-PER", "I-ORG", "I-LOC")
        self._state_set = {}
        self._transition_pro = {}
        self._emission_pro = {}
        self._init_state_set = {}
        self._hidden_state_set = {}
        self.training_data_path = KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_training_data.txt" if not training_data_path else training_data_path
        self.load_mode()
        for hidden_state in self._hidden_state_set:
            self.min_emission_pro = min([value for _, value in self._emission_pro[hidden_state].items()]) / 2

    def load_mode(self, state_set_save_path=None, transition_pro_save_path=None, emission_pro_save_path=None,
                  init_state_set_save_path=None, save_format="json"):
        def helper(file_path, save_format="json"):
            if save_format == "json":
                with open(file_path, encoding='utf-8') as f:
                    return json.load(f)

        state_set = KNLP_PATH + "/knlp/model/hmm/ner/state_set.json" if not state_set_save_path else state_set_save_path + "/state_set.json"
        transition_pro = KNLP_PATH + "/knlp/model/hmm/ner/transition_pro.json" if not transition_pro_save_path else transition_pro_save_path + "/transition_pro.json"
        emission_pro = KNLP_PATH + "/knlp/model/hmm/ner/emission_pro.json" if not emission_pro_save_path else emission_pro_save_path + "/emission_pro.json"
        init_state_set = KNLP_PATH + "/knlp/model/hmm/ner/init_state_set.json" if not init_state_set_save_path else init_state_set_save_path + "/init_state_set.json"
        self._state_set = helper(file_path=state_set)
        self._state_set["hidden_state"] = label_list(self.training_data_path)
        self._hidden_state_set = self._state_set["hidden_state"]
        self._transition_pro = helper(file_path=transition_pro)
        self._emission_pro = helper(file_path=emission_pro)
        self._init_state_set = helper(file_path=init_state_set)

    def viterbi(self, observe_seq, hidden_state_set=None, init_state_set=None, transition_pro=None, emission_pro=None,
                min_prob=3.14e-200):
        zero_check = 0
        if not hidden_state_set:
            hidden_state_set = self._hidden_state_set
        if not init_state_set:
            init_state_set = self._init_state_set
        if not transition_pro:
            transition_pro = self._transition_pro
        if not emission_pro:
            emission_pro = self._emission_pro
        viterbi_matrix = [{}]  # 每个timestep的几个概率大小，数组的index为timestep，里面的字典为概率值。可以想象为一个矩阵，横轴为timestep，纵轴为不同的概率值
        path = {}  # key 是当前的使整体概率最大的hidden state，value是一个数组，保存了路由到当前这个hidden state的，之前的所有的hidden state
        # 计算初始状态的概率分布，以及对应的路径， timestep = 1
        for hidden_state in hidden_state_set:
            viterbi_matrix[0][hidden_state] = init_state_set[hidden_state] * emission_pro[hidden_state].get(
                observe_seq[0], 0)
            if zero_check == 0:
                zero_check = viterbi_matrix[0][hidden_state]
            path[hidden_state] = [hidden_state]
        # 计算后续几个状态的分布以及对应的路径，timestep > 1
        if zero_check == 0:
            for hidden_state in hidden_state_set:
                viterbi_matrix[0][hidden_state] = min_prob
                path[hidden_state] = [hidden_state]
        for timestep in range(1, len(observe_seq)):
            viterbi_matrix.append({})
            new_path = {}
            for hidden_state in hidden_state_set:  # 循环遍历这个timestep可以采用的每一个hidden state
                # 这里这个就是我们求解viterbi算法需要使用的公式
                # 这里做的就是将上一个timestep的每个状态到达当前的状态的概率大小计算了一下，得到最大的那个状态以及对应的概率值
                # 注意max在对数组处理的时候会用数组的第一个元素进行处理，如果第一个值一样，会用第二个进行
                max_prob, arg_max_prob_hidden_state = max([(
                    viterbi_matrix[timestep - 1][hidden_state0] * transition_pro[hidden_state0].get(hidden_state,
                                                                                                    0) *  # noqa
                    emission_pro[hidden_state].get(observe_seq[timestep], self.min_emission_pro), hidden_state0) for
                    hidden_state0 in hidden_state_set if
                    viterbi_matrix[timestep - 1][
                        hidden_state0] > 0], default=(0.0, 'O'))  # 这个for循环 循环的是前一个timestep的所有hidden state. 使用min score 来应对未登录字
                viterbi_matrix[timestep][hidden_state] = max_prob  # 找到最大的之后，作为结果记录在viterbi矩阵中
                # 把新节点（hidden_state）添加到到达这个路径最大的那个hidden state对应的路径中
                new_path[hidden_state] = path[arg_max_prob_hidden_state] + [hidden_state]
            path = new_path

        # 需要对最后一个timestep的节点单独处理
        prob, state = max([(viterbi_matrix[-1][y], y) for y in self._end_set])

        return prob, path[state]

    def get_emission(self, char, emission_pro):
        if not emission_pro:
            emission_pro = self._emission_pro
        for key in emission_pro.keys():
            char_all = emission_pro[key].keys()
            if char in char_all:
                pass

    def bmes(self, sentence):
        # call viterbi to cut this sentence
        entity_list = label_list(self.training_data_path)
        prob, pos_list = self.viterbi(sentence, entity_list)
        # print("POS结果：" + str(pos_list))
        text_list = [s for s in sentence]
        self.cut_bmes(text_list, pos_list)

    def bios(self, sentence):
        # call viterbi to cut this sentence
        entity_list = label_list(self.training_data_path)
        prob, pos_list = self.viterbi(sentence, entity_list)
        # print("POS结果：" + str(pos_list))
        text_list = [s for s in sentence]
        self.cut_bio(text_list, pos_list)


if __name__ == '__main__':
    training_data_path = KNLP_PATH + '/knlp/data/msra_bios/train.bios'

    test = HMMInference(training_data_path)
    test_sen = "为贫困地区教育事业作贡献———访团中央书记处书记姜大明１９９６年，王玉梅又投资创办了全国最大的兔肉制品企业，并将公司更名为济南绿色兔业集团公司，集养殖、加工、销售于一体，形成了公司连基地、基地带农户的养殖新模式，成为全国最大的兔业综合基地。"
    test.bios(test_sen)
    print(test.get_sent())
    print(test.get_entity())
