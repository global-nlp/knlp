from knlp.NER.trie_seg.inference import TrieInference
from knlp.common.constant import KNLP_PATH
from knlp.NER.crf.inference import Inference
from knlp.NER.bilstm_crf.inference_ner import NerInference


class Pipeline:
    def __init__(self, type, input, model):
        self.type = type
        self.task = 'ner'
        if input: self.words = input
        self.data_path = ''
        self.model = model
        if self.type == 'inference':
            self.inference(self.model)
        elif self.type == 'train':
            self.train()
        else:
            print('only support inference or train method')

    def train(self, model, data_path):
        pass

    def inference(self, model):
        model_list = ['crf', 'trie', 'bilstm', 'bert_mrc', 'bert_tagger']
        if model not in model_list:
            print(f'only support model in {model_list}')
        else:
            if model == 'crf':
                self.crf_inference(self.words)
            elif model == 'trie':
                self.trie_inference(self.words)
            elif model == 'bilstm':
                self.bilstm_inference(self.words)
            elif model == 'bert_mrc':
                self.bert_mrc_inference(self.words)
            elif model == 'bert_tagger':
                self.bert_tagger_inference(self.words)

    def crf_inference(self, words):
        print("********crf_result********")
        test = Inference()
        CRF_MODEL_PATH = KNLP_PATH + "/knlp/model/crf/ner.pkl"

        print("读取数据...")
        to_be_pred = words

        test.spilt_predict(to_be_pred, CRF_MODEL_PATH)
        print("POS结果：" + str(test.label_prediction))
        print("模型预测结果：" + str(test.out_sentence))

    def trie_inference(self, words):
        print("********trie_result********")
        trieTest = TrieInference()
        print(trieTest.get_DAG(words, trieTest._trie))
        print(trieTest.knlp_seg(words))

    def bilstm_inference(self, words):
        print("********bilstm_result********")
        inference = NerInference()
        print(inference([words]))

    def bert_mrc_inference(self, words):
        pass

    def bert_tagger_inference(self, words):
        pass


if __name__ == '__main__':
    Pipeline(type='inference', model='crf', input='普林斯顿大学的阿格琉斯，是来自哪个国家的？')
    Pipeline(type='inference', model='trie', input='普林斯顿大学的阿格琉斯，是来自哪个国家的？')
    Pipeline(type='inference', model='bilstm', input='普林斯顿大学的阿格琉斯，是来自哪个国家的？')
