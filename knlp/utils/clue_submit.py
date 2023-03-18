import json
from knlp.seq_labeling.NER.trie_seg.inference import TrieInference
from knlp.common.constant import KNLP_PATH


class clue_submit:
    """
    这个类主要用作将trie预测结果转换为clue的提交格式。
    """
    def __init__(self):
        self.in_json_file = KNLP_PATH + r'\knlp\data\cluener_public\test.json'
        self.out_json_file = KNLP_PATH + r'\knlp\data\cluener_public\predict.json'
        self.in_json = open(self.in_json_file, encoding='utf-8')

    def eval_ner(self):
        out = open(self.out_json_file, 'w')
        for line in self.in_json.readlines():
            line = eval(line)
            tmp_dict = {}
            id = line["id"]
            text = line["text"]
            tmp_dict["id"] = int(id)
            tmp_dict["label"] = {}
            trieTest = TrieInference()
            list_word, list_label = trieTest.knlp_seg(text)
            for idx, word in enumerate(list_word):
                if list_label[idx] != 'O':
                    if word in tmp_dict["label"].keys():
                        tmp_dict["label"][list_label[idx]][word] = [[text.find(word), text.find(word) + len(word) - 1]]
                    else:
                        tmp_dict["label"][list_label[idx]] = {}
                        tmp_dict["label"][list_label[idx]][word] = [[text.find(word), text.find(word) + len(word) - 1]]
            j = json.dumps(tmp_dict, ensure_ascii=False)
            out.write(j)
            out.write('\n')
            # print(tmp_dict)
        out.close()


if __name__ == '__main__':
    # trieTest = TrieInference()
    # print(trieTest.get_DAG("你会和星级厨师一道先从巴塞罗那市中心兰布拉大道的laboqueria市场的开始挑选食材，", trieTest._trie))
    # print(trieTest.knlp_seg("销售冠军：辐射3-Bethesda"))
    evalner = clue_submit()

    # test_file = open(evalner.in_json_file, encoding='utf-8')
    # for line in test_file.readlines():
    #     words = line["text"]
    #     word_out, label_out = trieTest.knlp_seg(words)
    #     evalner.eval_ner()
    evalner.eval_ner()

