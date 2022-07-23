# -*-coding:utf-8-*-
import re

from knlp.seq_labeling.crf.crf import CRFModel
from knlp.common.constant import KNLP_PATH


class Inference:

    def __init__(self):
        self.label_prediction = []  # 预测标签序列
        self.out_sentence = []  # 预测结果序列

    def spilt_predict(self, in_put, file_path):
        """
        将输入序列分割为各个汉字团，依次送入输入的预训练模型中，返回各个汉字团的预测结果。
        """
        crf = CRFModel()

        re_zh, re_no_zh = re.compile("([\u4E00-\u9FA5]+)"), re.compile("[^a-zA-Z0-9+#\n]")  # 只对汉字做分词
        processed_sentence = re_zh.split(in_put)  # 按照汉字团进行分割
        crf_model = crf.load_model(file_path)

        for block in processed_sentence:
            if re_zh.match(block):  # 对汉字进行分词
                blocking = list(block)
                pred = [blocking]
                crf_pred = crf_model.test(pred)  # 预测
                pred = sum(pred, [])
                crf_pred = sum(crf_pred, [])
                self.out_sentence.extend(self.cut(pred, crf_pred))
                self.label_prediction.append(crf_pred)
            else:
                for char in re_no_zh.split(block):  # 把剩下的字符分出来
                    if char:
                        self.label_prediction.append(char)
                        self.out_sentence.append(char)
                    break

    def cut(self, sentence1, sentence2):
        """
        按照BIO标签做中文分词，切割句子。
        Args:
            sentence1: 文本序列
            sentence2: 标注序列

        Returns:

        """
        out_sent = []
        begin = 0
        for idx in range(len(sentence1)):
        #    print(sentence1[idx], sentence2[idx])
            if sentence2[idx][0] == 'B':
                begin = idx
            elif sentence2[idx][0] == 'I':
                idx += 1
                if idx == len(sentence1):
                    str = "".join(sentence1[begin:idx])
                    out_sent.append(str)
                    return out_sent
                elif sentence2[idx][0] == 'O' or sentence2[idx][0] == 'B':
                    str = "".join(sentence1[begin:idx])
                    out_sent.append(str)
                    begin = idx
            elif sentence2[idx][0] == 'O':
                out_sent.append(sentence1[idx])

        return out_sent


if __name__ == "__main__":
    test = Inference()
    CRF_MODEL_PATH = KNLP_PATH + "/knlp/model/crf/ner.pkl"

    print("读取数据...")
    to_be_pred = "斯坦福大学的阿格琉斯，正在前往爱琴海的路上。"

    test.spilt_predict(to_be_pred, CRF_MODEL_PATH)
    print("POS结果：" + str(test.label_prediction))
    print("模型预测结果：" + str(test.out_sentence))
