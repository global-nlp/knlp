# -*-coding:utf-8-*-
import pickle

from knlp.seq_labeling.crf.crf import CRFModel
from knlp.common.constant import KNLP_PATH


class Train:
    """
    这个类要实现针对不同task的训练语料数据的加载，构建并保存对应模型。
    这里的训练主要是训练出能够对输入拼音序列进行分割的模型，因此实现和中文分词相同，只不过读入的语料数据不同。
    """

    def __init__(self, data_path):

        self.training_data_path = data_path
        self.training_data = []
        self.model = CRFModel()

    def load_and_train(self):
        """
        读入数据后，存入words和tags两个列表中，传入train进行训练，返回训练后模型。
        """
        words = []
        tags = []

        with open(self.training_data_path, 'r', encoding='utf-8') as f:
            word_list = []
            tag_list = []
            for line in f:
                if line != '\n':
                    word, tag = line.strip('\n').split()
                    word_list.append(word)
                    tag_list.append(tag)
                else:
                    words.append(word_list)
                    tags.append(tag_list)
                    word_list = []
                    tag_list = []

        self.model.train(words, tags)

    def save_model(self, file_name):
        """用于保存训练模型"""
        with open(file_name, "wb") as f:
            pickle.dump(self.model, f)


if __name__ == "__main__":

    train_data_path = KNLP_PATH + "/knlp/data/pinyin_segment.txt"

    print("正在读入数据进行训练...")

    CRF_trainer = Train(train_data_path)
    CRF_trainer.load_and_train()

    print("正在保存模型...")

    CRF_trainer.save_model(KNLP_PATH + "/knlp/model/crf/pinyin.pkl")

    print("训练完成。")