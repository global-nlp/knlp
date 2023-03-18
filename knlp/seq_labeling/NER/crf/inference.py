# -*-coding:utf-8-*-
from knlp.seq_labeling.NER.Inference.Inference import NERInference
from knlp.seq_labeling.NER.trie_seg.ner_util import PostProcessTrie
from knlp.seq_labeling.crf.crf import CRFModel
from knlp.common.constant import KNLP_PATH

texts = [
    ('1945年', None),
    ('8月', None),
    ('爱琴海', 'scene'),
    ('斯坦福大学', 'organization'),
    ('斯坦福', 'organization'),
    ('雅典', 'organization')
]


class CRFInference(NERInference):

    def __init__(self):
        super().__init__()
        self.out_sentence = []  # 预测结果序列

    def spilt_predict(self, in_put, file_path):
        """
        将输入序列分割为各个汉字团，依次送入输入的预训练模型中，返回各个汉字团的预测结果。
        """
        crf = CRFModel()
        self.entity_set.clear()
        crf_model = crf.load_model(file_path)
        blocking = list(in_put)
        pred = [blocking]
        crf_pred = crf_model.test(pred)  # 预测
        pred = sum(pred, [])
        crf_pred = sum(crf_pred, [])
        self.cut_bio(pred, crf_pred)
        self.out_sentence.extend(self.get_sent())


if __name__ == "__main__":
    test = CRFInference()
    trieTree = PostProcessTrie()
    CRF_MODEL_PATH = KNLP_PATH + "/knlp/model/crf/ner.pkl"

    print("读取数据...")
    to_be_pred = '1945年8月斯坦福大学计算机学院阿克琉斯，想去雅典和爱琴海。中国与欧盟海军爱玩dota'

    test.spilt_predict(to_be_pred, CRF_MODEL_PATH)
    print("POS结果：" + str(test.get_tag()))
    print("模型预测结果：" + str(test.out_sentence))
    print("实体集合：" + str(test.get_entity()))

    trieTree.construct_text_dict([s for s in to_be_pred], sum(test.get_tag(), []))
    for word_tuples in texts:
        trieTree.insert(word_tuples[0], word_tuples[1], _from='dict')

    trieTree.post_soft_process(to_be_pred, test.entity_set)
    print(trieTree.get_entity())
    trieTree.post_hard_process(to_be_pred, test.entity_set)
    print(trieTree.get_entity())