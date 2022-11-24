import os
from knlp.seq_labeling.NER.trie_seg.inference import TrieInference
from knlp.common.constant import KNLP_PATH
from knlp.seq_labeling.NER.hmm.inference import HMMInference
from knlp.seq_labeling.NER.crf.inference import CRFInference
from knlp.seq_labeling.NER.bilstm_crf.inference_ner import BilstmInference
from knlp.seq_labeling.NER.bert.ner_inference import BertInference
from knlp.seq_labeling.NER.bert_mrc.predict import MRCNER_Inference
from knlp.seq_labeling.NER.trie_seg.ner_util import Later_process_Trie
from knlp.seq_labeling.bert.models.bert_for_ner import BertSoftmaxForNer
from knlp.seq_labeling.NER.ModelEval.eval_result import ModelEval
from knlp.seq_labeling.NER.bert.trainer import BERTTrain
from knlp.seq_labeling.NER.bert_mrc.train import MRCTrain
from knlp.seq_labeling.NER.bilstm_crf.train_bilstm_crf import TrainBiLSTMCRF
from knlp.seq_labeling.NER.hmm.train import HMMTrain
from knlp.seq_labeling.NER.crf.train import CRFTrain

model_list = ['hmm', 'crf', 'trie', 'bilstm', 'bert', 'mrc']

texts_add = [
    ('北京大学', 'ORG'),
    ('兰州', 'LOC')
]

texts_del = [
    ('北大', 'ORG')
]

class Pipeline:
    def __init__(self, model='all', data_sign=None, do_eval=False,
                 data_path=KNLP_PATH + '/knlp/data/msra_bios/train.bios',
                 vocab_path=KNLP_PATH + '/knlp/data/msra_bios/vocab.txt',
                 tagger_path=KNLP_PATH + '/knlp/data/msra_bios/', mrc_path=KNLP_PATH + '/knlp/data/msra_mrc',
                 add_dict=KNLP_PATH + '/knlp/data/user_dict/texts_add.txt',
                 del_dict=KNLP_PATH + '/knlp/data/user_dict/texts_del.txt',
                 from_user_txt=False):
        """
        :param model: 选择模型库中的某个模型，或全部模型
        :param data_sign: 指明数据集名称，主要对于bert的mrc方法中识别标签描述文件（msra.json）
        :param do_eval: 是否进行模型自我评估
        :param data_path: 数据集路径（具体到训练数据位置，用于hmm、crf、trie等等模型）
        :param vocab_path: 数据集vocab路径
        :param tagger_path: 用于bert序列标注的数据路径（到数据集目录位置即可，与data_path不同，不用具体到文件位置，上级文件夹即可）
        :param mrc_path: 用于bert阅读理解的数据路径
        :param from_user_txt: 是否来自用户添加字典
        """
        self.from_user_txt = from_user_txt
        self.del_dict = del_dict
        self.add_dict = add_dict
        self.task = data_sign

        if data_path:
            self.training_data_path = data_path
        if vocab_path:
            self.vocab_set_path = vocab_path
        if tagger_path:
            self.tagger_data_path = tagger_path
        if mrc_path:
            self.mrc_data_path = mrc_path

        # HMM 模型存储位置
        self.state_set_save_path = KNLP_PATH + "/knlp/model/hmm/ner"
        self.transition_pro_save_path = KNLP_PATH + "/knlp/model/hmm/ner"
        self.emission_pro_save_path = KNLP_PATH + "/knlp/model/hmm/ner"
        self.init_state_set_save_path = KNLP_PATH + "/knlp/model/hmm/ner"
        # CRF 模型存储位置
        self.crf_model_path = KNLP_PATH + "/knlp/model/crf/ner.pkl"
        # BiLSTM 模型存储位置
        self.model_path = KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/bilstm_crf_ner_msra.pkl"
        self.word2idx_path = KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/word2idx.json"
        self.tag2idx_path = KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/tag_json.json"
        # Trie 比较特殊，预处理时直接生成用于构建字典树的数据即可。预处理请看：knlp/seq_labeling/NER/preprocess.py
        # Bert-tagger 模型存储位置
        self.bert_tagger_save_path = KNLP_PATH + "/knlp/model/bert/tagger_"
        # Bert-mrc 模型存储位置
        self.bert_mrc_save_path = KNLP_PATH + '/knlp/model/bert/mrc_ner'

        # if input:
        #     self.words = input
        self.data_path = ''
        self.model = model
        # self.model_path_bert_tagger = KNLP_PATH + '/knlp/model/bert/output_modelbert'
        # self.model_path_bert_mrc = KNLP_PATH + '/knlp/model/bert/mrc_ner/checkpoint-63000.bin'
        self.do_eval = False
        self.trie = Later_process_Trie()

        if do_eval:
            self.do_eval = True
            self.dev = KNLP_PATH + '/knlp/data/msra_bios/val.bios'

        # if self.type == 'inference':
        #     self.inference(self.model)
        # elif self.type == 'train':
        #     self.train()
        # elif self.type == 'eval':
        #     self.do_eval = True
        #     self.dev = KNLP_PATH + '/knlp/data/msra_bios/val.bios'
        # elif self.type == 'all':
        #     self.do_eval = True
        #     self.dev = KNLP_PATH + '/knlp/data/msra_bios/val.bios'
        #     self.train()
        #     self.inference(self.model)
        # else:
        #     print('only support inference or train method')

    def train(self, model):
        model_list = ['hmm', 'crf', 'trie', 'bilstm', 'bert_tagger', 'bert_mrc']

        if model not in model_list and model != 'all':
            print(f'only support model in {model_list}')
        else:
            if model == 'hmm':
                self.hmm_train(self.state_set_save_path, self.transition_pro_save_path, self.emission_pro_save_path,
                               self.init_state_set_save_path)
            elif model == 'crf':
                self.crf_train(save_path=self.crf_model_path)
            elif model == 'trie':
                self.trie_train()
            elif model == 'bilstm':
                self.bilstm_train()
            elif model == 'bert_mrc':
                self.bert_mrc_train()
            elif model == 'bert_tagger':
                self.bert_tagger_train()
            elif model == 'all':
                self.hmm_train(self.state_set_save_path, self.transition_pro_save_path, self.emission_pro_save_path,
                               self.init_state_set_save_path)
                self.crf_train(save_path=self.crf_model_path)
                self.trie_train()
                self.bilstm_train()
                self.bert_mrc_train()
                self.bert_tagger_train()

    def bert_tagger_train(self):
        print('Bert-序列标注训练开始')
        BERTtrainer = BERTTrain(data_path=self.tagger_data_path, tokenizer_vocab=self.vocab_set_path,
                                data_sign=self.task, save_path=self.bert_tagger_save_path)
        BERTtrainer.run()
        print('Bert-序列标注训练结束')

    def bert_mrc_train(self):
        print('Bert-阅读理解训练开始')
        MRCtrainer = MRCTrain(data_path=self.mrc_data_path, data_sign=self.task, save_path=self.bert_mrc_save_path)
        MRCtrainer.run()
        print('Bert-阅读理解训练结束')

    def bilstm_train(self):
        print('BiLSTM训练开始')
        model_hyperparameters_dict = {
            "embedding_dim": 64,
            "hidden_dim": 64,
            "num_layers": 1
        }
        optimizer_hyperparameters_dict = {
            "lr": 0.01,
            "weight_decay": 1e-4
        }
        dataset_hyperparameters_dict = {
            "vocab_set_path": self.vocab_set_path,
            "training_data_path": self.training_data_path,
            "batch_size": 64,
            "shuffle": True
        }
        train = TrainBiLSTMCRF(model_hyperparameters=model_hyperparameters_dict,
                               optimizer_hyperparameters=optimizer_hyperparameters_dict,
                               dataset_hyperparameters=dataset_hyperparameters_dict)
        train.train(10)
        train.save(model_path=self.model_path, word2idx_path=self.word2idx_path, tag2idx_path=self.tag2idx_path)
        print('BiLSTM训练结束')

    def trie_train(self):
        print('Trie构建开始')
        trieTrain = TrieInference()
        print('Trie构建结束')

    def crf_train(self, save_path):
        print('CRF训练开始')
        CRF_trainer = CRFTrain(data_path=self.training_data_path)
        CRF_trainer.load_and_train()
        CRF_trainer.save_model(save_path)
        print('CRF训练结束')

    def hmm_train(self, state_set, trans_pro, emission_pro, init_state):
        print('HMM训练开始')
        vocab_set_path = self.vocab_set_path
        training_data_path = self.training_data_path

        state_set_save_path = state_set
        transition_pro_save_path = trans_pro
        emission_pro_save_path = emission_pro
        init_state_set_save_path = init_state

        trainer = HMMTrain(vocab_set_path=vocab_set_path, training_data_path=training_data_path)
        trainer.init_variable(vocab_set_path=vocab_set_path, training_data_path=training_data_path)
        trainer.build_model(state_set_save_path, transition_pro_save_path, emission_pro_save_path,
                            init_state_set_save_path)
        print('HMM训练结束')

    def inference(self, model, input, model_path_bert_tagger=None, model_path_bert_mrc=None):
        model_list = ['hmm', 'crf', 'trie', 'bilstm', 'bert_mrc', 'bert_tagger']

        self.words = input
        self.model_path_bert_tagger = model_path_bert_tagger if model_path_bert_tagger else KNLP_PATH + '/knlp/model/bert/output_modelbert'
        self.model_path_bert_mrc = model_path_bert_mrc if model_path_bert_mrc else KNLP_PATH + '/knlp/model/bert/mrc_ner/checkpoint-63000.bin'
        if model not in model_list and model != 'all':
            print(f'only support model in {model_list}')
        else:
            if model == 'hmm':
                self.hmm_inference(self.words)
            elif model == 'crf':
                self.crf_inference(self.words)
            elif model == 'trie':
                self.trie_inference(self.words)
            elif model == 'bilstm':
                self.bilstm_inference(self.words)
            elif model == 'bert_mrc':
                self.bert_mrc_inference(self.words, self.model_path_bert_mrc)
            elif model == 'bert_tagger':
                self.bert_tagger_inference(self.words, self.model_path_bert_tagger)
            elif model == 'all':
                self.hmm_inference(self.words)
                self.crf_inference(self.words)
                self.trie_inference(self.words)
                self.bilstm_inference(self.words)
                self.bert_tagger_inference(self.words, self.model_path_bert_tagger)
                self.bert_mrc_inference(self.words, self.model_path_bert_mrc)

    def hmm_inference(self, words):
        print("\n******** hmm_result ********\n")
        training_data_path = self.training_data_path
        test = HMMInference(training_data_path=training_data_path)
        test.bios(words)
        print("模型预测结果：" + str(test.out_sent))
        print("POS结果：" + str(test.tag_list))
        print("实体集合：" + str(test.get_entity()))

        if self.do_eval:
            self.eval_interpret('hmm')
        self.later_process_by_trie(words, sum(test.tag_list, []), test.get_entity())

    def crf_inference(self, words):
        print("\n******** crf_result ********\n")
        test = CRFInference()
        CRF_MODEL_PATH = KNLP_PATH + "/knlp/model/crf/ner.pkl"

        print("读取数据...")
        to_be_pred = words

        test.spilt_predict(to_be_pred, CRF_MODEL_PATH)
        print("POS结果：" + str(sum([], test.get_tag())))
        print("模型预测结果：" + str(test.get_sent()))
        print("实体集合：" + str(test.get_entity()))

        if self.do_eval:
            self.eval_interpret('crf')
        self.later_process_by_trie(words, sum(test.get_tag(), []), test.get_entity())

    def trie_inference(self, words):
        print("\n******** trie_result ********\n")
        trieTest = TrieInference()
        # use trie to finetune directly
        print(trieTest.get_DAG(words, trieTest._trie))
        print(trieTest.knlp_seg(words))
        print("实体集合：" + str(trieTest.get_entity()))

        if self.do_eval:
            self.eval_interpret('trie')
        self.later_process_by_trie(words, trieTest.get_tag(), trieTest.get_entity())

    def bilstm_inference(self, words):
        print("\n******** bilstm_result ********\n")
        inference = BilstmInference()
        print("模型预测结果：" + str(inference(words)))
        print("POS结果：" + str(inference.get_tag()))
        print("实体集合：" + str(inference.get_entity()))

        if self.do_eval:
            self.eval_interpret('bilstm')
        self.later_process_by_trie(words, sum(inference.get_tag(), []), inference.get_entity())

    def bert_tagger_inference(self, words, model_path):
        print("\n******** bert_result ********\n")
        inference = BertInference(task=self.task, log=False)
        model = BertSoftmaxForNer.from_pretrained(model_path)
        model.to('cpu')
        result = inference.predict(words, model)
        print("模型预测结果：" + str(result))
        print("POS结果：" + str(inference.get_tag()))
        print("实体集合：" + str(inference.get_entity()))
        # print(inference.run(words))
        if self.do_eval:
            self.eval_interpret('bert')
        self.later_process_by_trie(words, inference.get_tag(), inference.get_entity())

    def bert_mrc_inference(self, words, model_path):
        print("\n******** mrc_result ********\n")
        inference = MRCNER_Inference(mrc_data_path=self.mrc_data_path, tokenizer_vocab=self.vocab_set_path,
                                     data_sign=self.task, log=False)
        inference.config.saved_model = model_path
        inference.run(words)

        print("POS结果：" + str(inference.get_tag()))
        print("实体集合：" + str(inference.get_entity()))

        if self.do_eval:
            self.eval_interpret('mrc')
        self.later_process_by_trie(words, inference.get_tag(), inference.get_entity())

    def eval_interpret(self, model_1, model_2=None):
        self.do_eval = True
        self.dev = KNLP_PATH + '/knlp/data/msra_bios/val.bios'
        if self.do_eval:
            if not model_2:
                model_2 = model_1
                val_1 = ModelEval(self.dev, model=model_1, mrc_data_path=self.mrc_data_path,
                                  tokenizer_vocab=self.vocab_set_path, data_sign=self.task,
                                  tagger_path=self.model_path_bert_tagger, mrc_path=self.model_path_bert_mrc)
                val_1.evaluate()
            else:
                val_1 = ModelEval(self.dev, model=model_1, mrc_data_path=self.mrc_data_path,
                                  tokenizer_vocab=self.vocab_set_path, data_sign=self.task)
                val_1.evaluate()
                val_2 = ModelEval(self.dev, model=model_2, mrc_data_path=self.mrc_data_path,
                                  tokenizer_vocab=self.vocab_set_path, data_sign=self.task)
                val_2.evaluate()
            os.chdir(f"{KNLP_PATH}/knlp/seq_labeling/NER/interpretEval/")
            os.system(f"bash {KNLP_PATH}/knlp/seq_labeling/NER/interpretEval/run_task_ner.sh {model_1} {model_2}")
            os.chdir("./")
        else:
            print("To evaluate, set do_eval to True.")

    def later_process_by_trie(self, words, labels, entity_set, type=None):
        """
        :param words: 推理用的句子
        :param labels: 预测得到的标签
        :param entity_set: 预测获得的实体集合（这里加入实体集合，是因为标签数组只适用于非nested的方法）
        :param type: 后处理方式（软/硬）：软处理会保留所有预测得到的标；硬处理会在预测实体与词典中实体出现交叠时，保留词典中实体。
        """
        self.trie.construct_text_dict([s for s in words], labels)
        if self.from_user_txt:
            texts_add.clear()
            texts_del.clear()
            add = open(self.add_dict, 'r')
            for line in add.readlines():
                if line:
                    word, feature = line.strip().split(' ')
                    texts_add.append((word, feature))
            delete = open(self.del_dict, 'r')
            for line in delete.readlines():
                if line:
                    word, feature = line.strip().split(' ')
                    texts_add.append((word, feature))
        for word_tuples in texts_add:
            self.trie.insert(word_tuples[0], word_tuples[1], _from='dict')
        for word_del in texts_del:
            self.trie.del_word(word_del[0])
            if word_del in entity_set:
                entity_set.remove(word_del)
        if not type:
            self.trie.later_soft_process(words, entity_set)
            print("软处理结果：" + str(self.trie.get_entity()))
            self.trie.later_hard_process(words, entity_set)
            print("硬处理结果：" +str(self.trie.get_entity()))
        if type == 'soft':
            self.trie.later_soft_process(words, entity_set)
            print("软处理结果：" + str(self.trie.get_entity()))
        elif type == 'hard':
            self.trie.later_hard_process(words, entity_set)
            print("硬处理结果：" + str(self.trie.get_entity()))


if __name__ == '__main__':
    sentence = '毕业于北京大学的他，最爱读的书是《时间简史》。喜欢吃兰州拉面，曾任时间管理局局长。闲暇时喜欢玩csgo，对《星际穿越》赞叹有加。'
    pipe = Pipeline(data_sign='msra')
    pipe.inference(model='all', input=sentence)
    pipe.eval_interpret('hmm', 'crf')