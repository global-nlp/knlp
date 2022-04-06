# coding: utf-8
import json
import math

from knlp.common.constant import KNLP_PATH
from knlp.seq_labeling.crf.crf import CRFModel


class Inference:

    def __init__(self):

        self.pinyin_hanzi = {}
        self.start_state = {}
        self.emission_pro = {}
        self.transition_pro = {}

        self.label_prediction = []  # 预测标签序列
        self.out_sentence = []  # 预测结果序列
        self.crf = CRFModel()
        self.load_model()

    def load_model(self):
        def helper(file_path):
            with open(file_path) as f:
                return json.load(f)

        self.pinyin_hanzi = helper(KNLP_PATH + r'/knlp/data/pinyin_input_data/pinyin_hanzi.json')
        self.start_state = helper(KNLP_PATH + r'/knlp/data/pinyin_input_data/start_state.json')
        self.emission_pro = helper(KNLP_PATH + r'/knlp/data/pinyin_input_data/emission_pro.json')
        self.transition_pro = helper(KNLP_PATH + r'/knlp/data/pinyin_input_data/transition_pro.json')

    def init_state_set(self, state):
        """
        初始状态概率，增加default处理未登录字。
        """
        data = self.start_state['data']
        default = self.start_state['default']

        if state in data:
            probility = data[state]
        else:
            probility = default
        return float(probility)

    def get_emission(self, state, observation):
        """
        返回发射概率，同样可处理未登录字。
        """
        data = self.emission_pro['data']
        default = self.emission_pro['default']

        if state not in data:
            return float(default)
        else:
            probility = data[state]

        if observation not in probility:
            return float(default)
        else:
            return float(probility[observation])

    def get_transition(self, start_state, next_state):
        """
        返回转移概率。
        """
        data = self.transition_pro['data']
        default = self.transition_pro['default']

        if start_state not in data:
            return float(default)

        probility = data[start_state]

        if next_state in probility:
            return float(probility[next_state])

        if 'default' in probility:
            return float(probility['default'])

        return float(default)

    def get_states(self, observation):
        """
        用于返回拼音下所有可能的汉字，作为维特比迭代时的范围。
        """
        return [state for state in self.pinyin_hanzi[observation]]

    def viterbi(self, observations, min_prob=3.14e-200):
        viterbi_matrix = [{}]
        # 初始状态。
        hidden_state_set = self.get_states(observations[0])
        for hidden_state in hidden_state_set:
            __score = math.log(max(self.init_state_set(hidden_state), min_prob)) + \
                      math.log(max(self.get_emission(hidden_state, observations[0]), min_prob))

            __path = [hidden_state]

            viterbi_matrix[0][hidden_state] = []
            dp = (__score, __path)
            viterbi_matrix[0][hidden_state].append(dp)

        # t > 0 时刻维特比
        for time_step in range(1, len(observations)):
            cur_obs = observations[time_step]
            viterbi_matrix.append({})
            prev_states = hidden_state_set
            hidden_state_set = self.get_states(cur_obs)
            for hidden_state in hidden_state_set:
                viterbi_matrix[1][hidden_state] = []
                for hidden_state0 in prev_states:  # from y0(t-1) to y(t)
                    for item in viterbi_matrix[0][hidden_state0]:
                        __score = item[0] + math.log(max(self.get_transition(hidden_state0, hidden_state), min_prob)) + \
                                  math.log(max(self.get_emission(hidden_state, cur_obs), min_prob))
                        __path = item[1] + [hidden_state]
                        dp = (__score, __path)
                        viterbi_matrix[1][hidden_state].append(dp)  # 将得分与相对应的路径存储在viterbi_matrix中

        result = {}

        for last_state in viterbi_matrix[-1]:
            for item in viterbi_matrix[-1][last_state]:
                result[item[0]] = item[1]

        return sorted(result.items(), key=lambda item: item[0], reverse=True)

    def spilt_predict(self, in_put, file_path):
        """
        将输入序列分割为各个汉字团，依次送入预训练模型中，返回各个汉字团的预测结果。
        """
        crf_model = self.crf.load_model(file_path)
        blocking = list(in_put)
        pred = [blocking]
        crf_pred = crf_model.test(pred)
        self.out_sentence = self.cut(pred, crf_pred)
        crf_pred = sum(crf_pred, [])
        self.label_prediction = crf_pred

    def cut(self, sentence1, sentence2):
        """
        按照BEMS标签切割拼音团。
        """
        out_sent = []
        sen1 = sum(sentence1, [])
        sen2 = sum(sentence2, [])
        begin = 0

        for idx in range(len(sen1)):
            if sen2[idx] == 'B':
                begin = idx
            elif sen2[idx] == 'S':
                str = "".join(sen1[idx])
                out_sent.append(str)
            elif sen2[idx] == 'E':
                next = idx + 1
                str = "".join(sen1[begin:next])
                out_sent.append(str)
                begin = 0

        return out_sent


if __name__ == '__main__':
    test = Inference()

    CRF = CRFModel()
    CRF_MODEL_PATH = KNLP_PATH + "/knlp/model/crf/pinyin.pkl"

    print("读取数据...")
    to_be_pred = "dongtianlailechuntianyejiangdaolai"

    test.spilt_predict(to_be_pred, CRF_MODEL_PATH)
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
