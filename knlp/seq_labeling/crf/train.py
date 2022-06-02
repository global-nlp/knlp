# -*-coding:utf-8-*-
import pickle
import sys

from knlp.seq_labeling.crf.crf import CRFModel
from knlp.common.constant import KNLP_PATH
from knlp.utils.util import get_model_crf_hanzi_file


class Train:
    """
    这个类要实现针对不同task的训练语料数据的加载，构建并保存对应模型。
    """

    def __init__(self, data_path=None):
        """
        Args:
            data_path:训练语料数据路径
        初始化训练模型。
        """
        self.training_data_path = data_path
        self.training_data = []
        self.model = CRFModel()
        if data_path:
            self.init_variable(training_data_path=data_path)

    def init_variable(self, training_data_path=None):
        self.training_data_path = KNLP_PATH + "/knlp/data/hanzi_segment.txt" if not training_data_path else training_data_path

        with open(self.training_data_path, encoding='utf-8') as f:
            self.training_data = f.readlines()

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

    def save_model(self, model_save_path):
        """
        Args:
            model_save_path:模型存储的地址
        用于保存训练模型
        """
        with open(model_save_path, "wb") as f:
            pickle.dump(self.model, f)


if __name__ == "__main__":

    args = sys.argv
    train_data_path = KNLP_PATH + "/knlp/data/hanzi_segment.txt"

    if len(args) > 1:
        train_data_path = args[1]

    print("正在读入数据进行训练...")

    CRF_trainer = Train(data_path=train_data_path)
    CRF_trainer.load_and_train()

    print("正在保存模型...")

    CRF_trainer.save_model(get_model_crf_hanzi_file())

    print("训练完成。")
