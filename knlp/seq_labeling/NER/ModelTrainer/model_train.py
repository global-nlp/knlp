from knlp.common.constant import KNLP_PATH, model_list
from knlp.seq_labeling.NER.bert.trainer import BERTTrain
from knlp.seq_labeling.NER.bert_mrc.train import MRCTrain
from knlp.seq_labeling.NER.bilstm_crf.train_bilstm_crf import TrainBiLSTMCRF
from knlp.seq_labeling.NER.hmm.train import HMMTrain
from knlp.seq_labeling.NER.crf.train import CRFTrain
from knlp.seq_labeling.NER.trie_seg.inference import TrieInference


class ModelTrainer:
    def __init__(self, data_path, vocab_path, tagger_path, mrc_path, model, data_sign):
        """
        :param data_path: 数据集路径（具体到训练数据位置，用于hmm、crf、trie等等模型）
        :param vocab_path: 数据集vocab路径
        :param tagger_path: 用于bert序列标注的数据路径（到数据集目录位置即可，与data_path不同，不用具体到文件位置，上级文件夹即可）
        :param mrc_path: 用于bert阅读理解的数据路径
        :param model: 选择模型库中的某个模型，或全部模型
        :param data_sign: 指明数据集名称，主要对于bert的mrc方法中识别标签描述文件（msra.json）
        """
        self.training_data_path = data_path
        self.vocab_set_path = vocab_path
        self.tagger_data_path = tagger_path
        self.mrc_data_path = mrc_path
        self.state_set_save_path = KNLP_PATH + "/knlp/model/hmm/ner"
        self.transition_pro_save_path = KNLP_PATH + "/knlp/model/hmm/ner"
        self.emission_pro_save_path = KNLP_PATH + "/knlp/model/hmm/ner"
        self.init_state_set_save_path = KNLP_PATH + "/knlp/model/hmm/ner"
        self.model = model
        self.task = data_sign
        self.model_list = model_list
        if not data_path:
            self.training_data_path = KNLP_PATH + '/knlp/data/bios_clue'
        if not mrc_path:
            self.mrc_data_path = KNLP_PATH + '/knlp/data/mrc/clue_mrc'

    def train(self):
        if self.model not in self.model_list and self.model != 'all':
            print(f'only support model in {self.model_list}')
        else:
            if self.model == 'hmm':
                self.hmm_train(self.state_set_save_path, self.transition_pro_save_path, self.emission_pro_save_path, self.init_state_set_save_path)
            elif self.model == 'crf':
                self.crf_train(save_path=KNLP_PATH + "/knlp/model/crf/ner.pkl")
            elif self.model == 'trie':
                self.trie_train()
            elif self.model == 'bilstm':
                self.bilstm_train()
            elif self.model == 'bert_mrc':
                self.bert_mrc_train()
            elif self.model == 'bert_tagger':
                self.bert_tagger_train()
            elif self.model == 'all':
                self.hmm_train(self.state_set_save_path, self.transition_pro_save_path, self.emission_pro_save_path, self.init_state_set_save_path)
                self.crf_train(save_path=KNLP_PATH + "/knlp/model/crf/ner.pkl")
                self.trie_train()
                self.bilstm_train()
                self.bert_mrc_train()
                self.bert_tagger_train()

    def your_model_train(self):
        """
        example:
        print('your_model_name-序列标注训练开始')
        YourModelTrainer = YourModelTrain(**params)
        YourModelTrainer.run(**params)
        print('your_model_name-序列标注训练结束')
        """
        pass

    def bert_tagger_train(self):
        print('Bert-序列标注训练开始')
        BERTtrainer = BERTTrain(data_path=self.tagger_data_path, tokenizer_vocab=self.vocab_set_path, data_sign=self.task)
        BERTtrainer.run()
        print('Bert-序列标注训练结束')

    def bert_mrc_train(self):
        print('Bert-阅读理解训练开始')
        MRCtrainer = MRCTrain(data_path=self.mrc_data_path, data_sign=self.task)
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
        model_path = KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/bilstm_crf_ner_msra.pkl"
        word2idx_path = KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/word2idx.json"
        tag2idx_path = KNLP_PATH + "/knlp/nn/bilstm_crf/model_bilstm_crf/tag_json.json"
        train = TrainBiLSTMCRF(model_hyperparameters=model_hyperparameters_dict,
                               optimizer_hyperparameters=optimizer_hyperparameters_dict,
                               dataset_hyperparameters=dataset_hyperparameters_dict)
        train.train(10)
        train.save(model_path=model_path, word2idx_path=word2idx_path, tag2idx_path=tag2idx_path)
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


if __name__ == '__main__':
    for model in ['hmm', 'crf', 'trie', 'bilstm',  'bert_tagger', 'bert_mrc']:
        test = ModelTrainer(data_path=KNLP_PATH + '/knlp/data/msra_bios/train.bios',
                       vocab_path=KNLP_PATH + '/knlp/data/cluener_public/cluener_vocab.txt',
                       mrc_path=KNLP_PATH + '/knlp/data/msra_mrc', model=model)
        test.train()
