from knlp.common.constant import KNLP_PATH, model_list
from knlp.seq_labeling.classification.bert.trainer import BertTrain
from knlp.nn.textcnn.train_textcnn import TrainTextCNN
from knlp.seq_labeling.classification.beyas.beyas_train import beyas_train


class ModelTrainer(PipeTrainer):
    def __init__(self, data_path, vocab_path, model):
        """
        :param data_path: 数据集路径（具体到训练数据位置，用于hmm、crf、trie等等模型）
        :param vocab_path: 数据集vocab路径
        :param model: 选择模型库中的某个模型，或全部模型
        """
        super().__init__()
        self.training_data_path = data_path
        self.vocab_set_path = vocab_path
        self.clf_model_path = KNLP_PATH + "/knlp/model/beyas/classification"
        self.tf_model_path = KNLP_PATH + "/knlp/model/beyas/classification"
        self.model = model
        self.model_list = class_model_list
        if not data_path:
            self.training_data_path = KNLP_PATH + '/knlp/data/class_clue'

    def train(self):
        if self.model not in self.model_list and self.model != 'all':
            print(f'only support model in {self.model_list}')
        else:
            if self.model == 'bert':
                self.bert_train()
            elif self.model == 'textcnn':
                self.textcnn_train(model_save_path=KNLP_PATH + "/knlp/model/classification/textcnn.pkl",
                                   word2idx_path=KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_word2idx.json",
                                   label2idx_path=KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_label2idx.json")
            elif self.model == 'beyas':
                self.beyas_train(clf_model_path=self.clf_model_path, tf_model_path=self.tf_model_path)
            elif self.model == 'all':
                self.bert_train()
                self.textcnn_train(model_save_path=KNLP_PATH + "/knlp/model/classification/textcnn.pkl",
                                   word2idx_path=KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_word2idx.json",
                                   label2idx_path=KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_label2idx.json")
                self.beyas_train(clf_model_path=self.clf_model_path, tf_model_path=self.tf_model_path)

    def your_model_train(self):
        """
        example:
        print('your_model_name-分类训练开始')
        YourModelTrainer = YourModelTrain(**params)
        YourModelTrainer.run(**params)
        print('your_model_name-分类训练结束')
        """
        pass

    def bert_train(self):
        print('Bert-文本分类训练开始')
        BertTrainer = BertTrain(data_path=self.tagger_data_path, tokenizer_vocab=self.vocab_set_path)
        BertTrainer.run()
        print('Bert-文本分类训练结束')

    def textcnn_train(self, model_save_path, word2idx_path, label2idx_path):
        kwargs = {
            "dataset_hyperparameters": {
                "vocab_set_path": self.vocab_set_path,
                # "training_data_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_eval_9988.txt",
                "training_data_path": self.training_data_path,
                # "eval_data_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_eval_9988.txt",
                "tokenizer": jieba.lcut,
                "shuffle": True,
                "batch_size": 64,
                "max_length": 150
            },
            "optimizer_hyperparameters": {
                "lr": 0.01,
                "weight_decay": 1e-4
            },
            "model_hyperparameters": {
                "n_filters": 100,
                "filter_sizes": [3, 4, 5]
            },
            # "non_static_word2vec_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_word2vec_300d_20509.txt",
            "static_word2vec_path": KNLP_PATH + "/knlp/nn/textcnn/data_textcnn/text_classification_weibo_word2vec_100d_22770.txt",

        }
        save_kwargs = {
            "model_path": model_save_path,
            "word2idx_path": word2idx_path,
            "label2idx_path": label2idx_path,
        }
        print("Textcnn-文本分类训练开始")
        train = TrainTextCNN(**kwargs)
        train.train(5)
        train.save(**save_kwargs)
        print("Textcnn-文本分类结束")

    def beyas_train(self, clf_model_path, tf_model_path):
        print("Beyas-文本分类开始")
        beyas = beyas_train(file_path=self.train_data, clf_model_path=clf_model_path, tf_model_path=tf_model_path)
        train_datas, train_labels = beyas.load_data()
        beyas.train_model(datas=train_datas, labels=train_labels)
        print("Beyas-文本分类结束")

if __name__ == '__main__':
    for model in ['bert', 'beyas', 'textcnn']:
        test = ModelTrainer(data_path=KNLP_PATH + '/knlp/data/msra_bios/train.bios',
                            vocab_path=KNLP_PATH + '/knlp/data/cluener_public/cluener_vocab.txt',
                             model=model)
        test.train()

