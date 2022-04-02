# !/usr/bin/python
# -*- coding:UTF-8 -*-

import json
import sys
from collections import defaultdict

from knlp.common.constant import KNLP_PATH


class Train:

    def __init__(self, vocab_set_path=None, training_data_path=None, test_data_path=None):

        self._state_set = {}
        self._transition_pro = {}
        self._emission_pro = {}
        self._init_state_set = {}
        self.vocab_set_path = ""
        self.training_data_path = ""
        self.vocab_data = []
        self.training_data = []
        if vocab_set_path and training_data_path:
            self.init_variable(vocab_set_path=vocab_set_path, training_data_path=training_data_path,
                               test_data_path=test_data_path)

    def init_variable(self, vocab_set_path=None, training_data_path=None, test_data_path=None):
        self.vocab_set_path = KNLP_PATH + "/knlp/data/hmm-ner_data/train/cluener_vocab.txt" if not vocab_set_path else vocab_set_path
        self.training_data_path = KNLP_PATH + "/knlp/data/hmm-ner_data/train/cluener_training_data.txt" if not training_data_path else training_data_path
        with open(self.vocab_set_path, encoding='utf-8') as f:
            self.vocab_data = f.readlines()

        with open(self.training_data_path, encoding='utf-8') as f:
            self.training_data = f.readlines()

    @property
    def state_set(self):
        self.set_state()
        return self._state_set

    @property
    def transition_pro(self):
        self.set_transition_pro()
        return self._transition_pro

    @property
    def emission_pro(self):
        self.set_emission_pro()
        return self._emission_pro

    @property
    def init_state_set(self):
        self.set_init_state_set()
        return self._init_state_set

    def set_state(self):
        self._state_set["hidden_state"] = ["O", "B-add", "I-add", "B-boo", "I-boo", "B-com", "I-com", "B-gam", "I-gam", "B-gov", "I-gov", "B-mov", "I-mov", "B-nam", "I-nam", "B-org", "I-org", "B-pos", "I-pos", "B-sce", "I-sce"]
        self._state_set["observation_state"] = []
        for line in self.vocab_data:
            self._state_set["observation_state"].append(line.strip())

    def set_transition_pro(self):

        count_dict = {
            "O": defaultdict(int),
            "B-add": defaultdict(int),
            "I-add": defaultdict(int),
            "B-boo": defaultdict(int),
            "I-boo": defaultdict(int),
            "B-com": defaultdict(int),
            "I-com": defaultdict(int),
            "B-gam": defaultdict(int),
            "I-gam": defaultdict(int),
            "B-gov": defaultdict(int),
            "I-gov": defaultdict(int),
            "B-mov": defaultdict(int),
            "I-mov": defaultdict(int),
            "B-nam": defaultdict(int),
            "I-nam": defaultdict(int),
            "B-org": defaultdict(int),
            "I-org": defaultdict(int),
            "B-pos": defaultdict(int),
            "I-pos": defaultdict(int),
            "B-sce": defaultdict(int),
            "I-sce": defaultdict(int)
        }
        for idx, line in enumerate(self.training_data):
            line = line.strip()
            total_lines = len(self.training_data)
            if not line:
                continue
            line = line.strip().split("\t")  # 获取到当前正在统计的那个标签
            if (idx + 1) < total_lines:
                next_line = self.training_data[idx + 1].strip()
            else:
                continue
            if not next_line:
                continue
            next_line = self.training_data[idx + 1].strip().split("\t")  # 获取下一个标签
            count_dict[line[-1]][next_line[-1]] += 1
        for start_label, end_labels in count_dict.items():

            self._transition_pro[start_label] = {}
            cnt_sum = sum(list(end_labels.values()))
            for end_label, count in end_labels.items():
                self._transition_pro[start_label][end_label] = count / cnt_sum

    def set_emission_pro(self):
        """
        统计获取发射概率

        Returns:

        """
        count_dict = {
            "O": defaultdict(int),
            "B-add": defaultdict(int),
            "I-add": defaultdict(int),
            "B-boo": defaultdict(int),
            "I-boo": defaultdict(int),
            "B-com": defaultdict(int),
            "I-com": defaultdict(int),
            "B-gam": defaultdict(int),
            "I-gam": defaultdict(int),
            "B-gov": defaultdict(int),
            "I-gov": defaultdict(int),
            "B-mov": defaultdict(int),
            "I-mov": defaultdict(int),
            "B-nam": defaultdict(int),
            "I-nam": defaultdict(int),
            "B-org": defaultdict(int),
            "I-org": defaultdict(int),
            "B-pos": defaultdict(int),
            "I-pos": defaultdict(int),
            "B-sce": defaultdict(int),
            "I-sce": defaultdict(int)
        }
        for line in self.training_data:
            if not line.strip():
                continue
            line = line.strip().split("\t")
            count_dict[line[-1]][line[0]] += 1
        for hidden_state, observation_states in count_dict.items():
            self._emission_pro[hidden_state] = {}
            cnt_sum = sum(list(observation_states.values()))
            for observation_state, count in observation_states.items():
                self._emission_pro[hidden_state][observation_state] = count / cnt_sum

    def set_init_state_set(self):
        """

        当这个字开头的时候，有多大的概率是哪个标签
        {WORD: {LABEL: PRO}}
        Returns:

        """
        count_dict = {
            "O": 0,
            "B-add": 0,
            "I-add": 0,
            "B-boo": 0,
            "I-boo": 0,
            "B-com": 0,
            "I-com": 0,
            "B-gam": 0,
            "I-gam": 0,
            "B-gov": 0,
            "I-gov": 0,
            "B-mov": 0,
            "I-mov": 0,
            "B-nam": 0,
            "I-nam": 0,
            "B-org": 0,
            "I-org": 0,
            "B-pos": 0,
            "I-pos": 0,
            "B-sce": 0,
            "I-sce": 0
        }
        for line in self.training_data:
            if not line.strip():
                continue
            line = line.strip().split("\t")
            count_dict[line[-1]] += 1
        cnt_sum = sum(list(count_dict.values()))
        for start_label, cnt in count_dict.items():
            self._init_state_set[start_label] = cnt / cnt_sum

    @staticmethod
    def save_model(file_path, data, format="json"):
        if format == "json":
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)

    def build_model(self, state_set_save_path=None, transition_pro_save_path=None, emission_pro_save_path=None,
                    init_state_set_save_path=None):
        """
        依次运行以上的几个函数，然后将获取到的结果保存下来

        Returns:
        """
        state_set = KNLP_PATH + "/knlp/model/hmm-ner/state_set.json" if not state_set_save_path else state_set_save_path + "\\state_set.json"
        transition_pro = KNLP_PATH + "/knlp/model/hmm-ner/transition_pro.json" if not transition_pro_save_path else transition_pro_save_path + "\\transition_pro.json"
        emission_pro = KNLP_PATH + "/knlp/model/hmm-ner/emission_pro.json" if not emission_pro_save_path else emission_pro_save_path + "\\emission_pro.json"
        init_state_set = KNLP_PATH + "/knlp/model/hmm-ner/init_state_set.json" if not init_state_set_save_path else init_state_set_save_path + "/init_state_set.json"
        self.save_model(file_path=state_set, data=self.state_set)
        self.save_model(file_path=transition_pro, data=self.transition_pro)
        self.save_model(file_path=emission_pro, data=self.emission_pro)
        self.save_model(file_path=init_state_set, data=self.init_state_set)


if __name__ == '__main__':
    # input path for vocab and training data
    args = sys.argv
    vocab_set_path = None
    training_data_path = None

    if len(args) > 1:
        vocab_set_path = args[1]
        training_data_path = args[2]

    a = Train(vocab_set_path=vocab_set_path, training_data_path=training_data_path)
    a.init_variable()
    a.build_model()
