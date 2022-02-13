# -*-coding:utf-8-*-
import re

from crf import load_model
from knlp.common.constant import KNLP_PATH
CRF_MODEL_PATH = KNLP_PATH+"/knlp/model/crf/crf.pkl"

def spilt_predict(str1):

    re_zh, re_no_zh = re.compile("([\u4E00-\u9FA5]+)"), re.compile("[^a-zA-Z0-9+#\n]")  # 只对汉字做分词
    processed_sentence = re_zh.split(str1)  # 按照汉字团进行分割
    out_sent = []
    crf_model = load_model(CRF_MODEL_PATH)

    for block in processed_sentence:
        if re_zh.match(block):  # 对汉字进行分词
            blocking = list(block)
            pred = [blocking]
            crf_pred = crf_model.test(pred)  # 预测
            out_sent.extend(cut(pred, crf_pred))
        else:
            for char in re_no_zh.split(block):  # 把剩下的分出来
                if(block):out_sent.append(block)
                break

    return out_sent

def cut(sentence1, sentence2):

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

    print("读取数据...")
    to_be_pred = "《美国夫人》是大魔王首度担任主演、执行制片人的美剧。也是鱼叔2020年看过最震撼的剧。它展现了美国 1970 年代第二波女权运动是如何风起云涌，又是如何偃旗息鼓的。"
    predict = spilt_predict(to_be_pred)
    print(predict)