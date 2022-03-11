# -*-coding:utf-8-*-
import re
import os

from crf import CRFModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../.."


def spilt_predict(in_put, file_path):
    """
    将输入序列分割为各个汉字团，依次送入预训练模型中，返回各个汉字团的预测结果。
    """
    crf = CRFModel()

    re_zh, re_no_zh = re.compile("([\u4E00-\u9FA5]+)"), re.compile("[^a-zA-Z0-9+#\n]")  # 只对汉字做分词
    processed_sentence = re_zh.split(in_put)  # 按照汉字团进行分割
    out_sent = []
    label_prediction = []
    crf_model = crf.load_model(file_path)

    for block in processed_sentence:
        if re_zh.match(block):  # 对汉字进行分词
            blocking = list(block)
            pred = [blocking]
            crf_pred = crf_model.test(pred)  # 预测
            out_sent.extend(cut(pred, crf_pred))
            crf_pred = sum(crf_pred, [])
            label_prediction.append(crf_pred)
        else:
            for char in re_no_zh.split(block):  # 把剩下的字符分出来
                if block:
                    label_prediction.append(block)
                    out_sent.append(block)
                break

    return label_prediction, out_sent


def cut(sentence1, sentence2):
    """
    按照BEMS标签做中文分词，切割句子。
    """
    out_sen = []
    sen1 = sum(sentence1, [])
    sen2 = sum(sentence2, [])
    begin = 0

    for idx in range(len(sen1)):
        if sen2[idx] == 'B':
            begin = idx
        elif sen2[idx] == 'S':
            str = "".join(sen1[idx])
            out_sen.append(str)
        elif sen2[idx] == 'E':
            next = idx + 1
            str = "".join(sen1[begin:next])
            out_sen.append(str)
            begin = 0

    return out_sen


if __name__ == "__main__":

    CRF = CRFModel()
    CRF_MODEL_PATH = BASE_DIR + "/knlp/model/crf/crf.pkl"
    print("读取数据...")
    to_be_pred = "我来自中国，我是炎黄子孙。"
    predict, result = spilt_predict(to_be_pred, CRF_MODEL_PATH)
    print("POS结果：" + str(predict))
    print("模型预测结果：" + str(result))
