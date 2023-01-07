from tqdm import tqdm

from knlp.common.constant import KNLP_PATH, delimiter, model_list
from knlp.seq_labeling.NER.bert.ner_inference import BertInference
from knlp.seq_labeling.NER.bert_mrc.predict import MRCNER_Inference
from knlp.seq_labeling.NER.bilstm_crf.inference_ner import BilstmInference
from knlp.seq_labeling.NER.crf.inference import CRFInference
from knlp.seq_labeling.NER.hmm.inference import HMMInference
from knlp.seq_labeling.NER.trie_seg.inference import TrieInference
from knlp.seq_labeling.bert.models.bert_for_ner import BertSoftmaxForNer


def generate_file(dev, pred_list, output):
    """
    dev: 用于评估的文件
    pred_list: 将用于评估的句子预测获得标签列表
    output: 新产生的文件，一列是dev文件中的汉字，一列是预测的标签
    """
    pred_all = sum(pred_list, [])
    f = open(dev, 'r', encoding='utf-8')
    out = open(output, 'w', encoding='utf-8')
    devv = []
    for line in f.readlines():
        if line != '\n':
            sentence = line.strip('\n')
            devv.append(sentence)
        else:
            devv.append(line)
    for index, piece in enumerate(devv):
        if piece != '\n':
            str = piece + '\t' + pred_all[index] + '\n'
            out.write(str)
        else:
            out.write('\n')


def construct_sent(dev):
    """
    从dev文件生成用于预测的句子
    """
    f = open(dev, 'r', encoding='utf-8')
    sents = []
    str = ''
    for line in f.readlines():
        if line != '\n':
            text, label = line.strip('\n').split(delimiter)
            str += text
        else:
            sents.append(str)
            str = ''
    return sents


def tab2blank(file1, file2):
    """
    将file1中的制表符分隔改为空格分隔（仅为了服务于interpret这个仓库，后续重构为自己的评估之后就不需要了）
    """
    file = open(file1, 'r', encoding='utf-8')
    out = open(file2, 'w', encoding='utf-8')
    for line in file.readlines():
        if line != '\n':
            str = line.replace('\t', ' ')
            out.write(str)
        else:
            out.write(line)


class ModelEval:
    def __init__(self, dev_path, model, mrc_data_path=None, tokenizer_vocab=None, data_sign=None, tagger_path=None, mrc_path=None):
        """
        :param dev_path: eval数据集路径
        :param model: 选择模型库中的某个模型，或全部模型
        :param mrc_data_path: 用于bert阅读理解的数据路径
        :param tokenizer_vocab: 数据集vocab路径
        :param data_sign: 指明数据集名称，主要对于bert的mrc方法中识别标签描述文件（msra.json）
        :param tagger_path: 用于bert序列标注的数据路径（到数据集目录位置即可，与data_path不同，不用具体到文件位置，上级文件夹即可）
        :param mrc_path: 用于bert阅读理解的数据路径
        """
        self.model = model
        self.dev_path = dev_path
        self.mrc_data_path = mrc_data_path
        self.vocab = tokenizer_vocab
        self.task = data_sign
        self.model_path_bert_tagger = tagger_path
        self.model_path_bert_mrc = mrc_path
        self.model_list = model_list
        if model not in self.model_list:
            print("Model name required!")
        if not dev_path:
            print("dev_file required!")
        self.for_pred = construct_sent(self.dev_path)
        self.pred_list = []

    def evaluate(self):
        out = KNLP_PATH + f'/knlp/seq_labeling/NER/interpretEval/data/ner/clue/results/{self.model}_eval.txt'

        if self.model == 'hmm':
            self.__hmm()
        elif self.model == 'crf':
            self.__crf()
        elif self.model == 'trie':
            self.__trie()
        elif self.model == 'bilstm':
            self.__bilstm()
        elif self.model == 'bert':
            self.__bert_tagger()
        elif self.model == 'mrc':
            self.__mrc()
        elif self.model == 'your_new_model_here':
            self.__your_new_model_here()

        generate_file(self.dev_path, self.pred_list, out)
        out_blank = KNLP_PATH + f'/knlp/seq_labeling/NER/interpretEval/data/ner/clue/results/{self.model}_eval_b.txt'
        tab2blank(out, out_blank)

    def __hmm(self):
        training_data_path = self.dev_path
        test = HMMInference(training_data_path)
        for sentence in tqdm(self.for_pred):
            test.bios(sentence)
            res = sum(test.get_tag(), [])
            self.pred_list.append(res)
            test.tag_list.clear()
            self.pred_list.append(['\n'])

    def __crf(self):
        test = CRFInference()
        CRF_MODEL_PATH = KNLP_PATH + "/knlp/model/crf/ner.pkl"
        for sentence in tqdm(self.for_pred):
            test.spilt_predict(sentence, CRF_MODEL_PATH)
            res = sum(test.get_tag(), [])
            test.tag_list.clear()
            self.pred_list.append(res)
            self.pred_list.append(['\n'])

    def __trie(self):
        trieTest = TrieInference()
        for sentence in tqdm(self.for_pred):
            trieTest.knlp_seg(sentence)
            res = trieTest.get_tag()
            self.pred_list.append(list(res))
            self.pred_list.append(['\n'])
            trieTest.tag_list.clear()

    def __bilstm(self):
        inference = BilstmInference()
        for sentence in tqdm(self.for_pred):
            res = inference(sentence)
            tag = inference.get_tag()
            tag = sum(tag, [])
            self.pred_list.append(tag)
            self.pred_list.append(['\n'])
            inference.tag_list.clear()

    def __bert_tagger(self):
        inference = BertInference(task=self.task)
        model = BertSoftmaxForNer.from_pretrained(self.model_path_bert_tagger)
        model.to('cpu')
        for sentence in tqdm(self.for_pred):
            inference.predict(sentence, model)
            result = inference.get_tag()
            self.pred_list.append(list(result))
            self.pred_list.append(['\n'])

    def __mrc(self):
        init_tag = []
        for sentence in tqdm(self.for_pred):
            test = MRCNER_Inference(mrc_data_path=self.mrc_data_path, tokenizer_vocab=self.vocab, data_sign=self.task)
            test.config.saved_model = self.model_path_bert_mrc
            init_tag = ['O' for _ in range(len(sentence))]
            test.run(sentence)
            res = test.get_chunks()
            for contain in res:
                for piece in contain:
                    union = ''.join(piece[0])
                    label = piece[2]
                    begin = sentence.lower().find(union)
                    end = begin + len(union) - 1
                    init_tag[begin] = 'B' + '-' + label
                    middle_tags = [('I' + '-' + label) for _ in range(end - begin)]
                    init_tag[begin + 1:end + 1] = middle_tags

            self.pred_list.append(list(init_tag))
            self.pred_list.append(['\n'])

            init_tag.clear()

    def __your_new_model_here(self):
        pass
