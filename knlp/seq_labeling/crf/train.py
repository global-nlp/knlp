# -*-coding:utf-8-*-
import os
from crf import crf_train

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../.."

def load_data(data_path):

    words = []
    tags = []

    with open(data_path, 'r', encoding='utf-8') as f:
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

    return words, tags


if __name__ == "__main__":

    train_data_path = BASE_DIR + "/knlp/data/pku.txt"
    train_words, train_tags = load_data(train_data_path)

    print("正在训练CRF模型...")

    crf_pred = crf_train((train_words, train_tags))
