from knlp.common.constant import KNLP_PATH, class_model_list
from knlp.nn.textcnn.inference_textcnn import InferenceTextCNN
import jieba
from knlp.Pipeline.pipeline import Pipeline
from knlp.seq_labeling.classification.ModelTrainer.model_train import ModelTrainer
from knlp.seq_labeling.classification.bert.inference import bertinference
import torch
import argparse




class ClassificationPipeline(Pipeline):

    def __init__(self, type, data_path=KNLP_PATH + '/knlp/data/bios_clue/train.txt',
                 dev_path=KNLP_PATH + '/knlp/data/clue/val.txt',
                 vocab_path=KNLP_PATH + '/knlp/data/clue/vocab.txt',
                 word2idx_path=KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_word2idx.json",
                 label2idx_path=KNLP_PATH + "/knlp/nn/textcnn/model_textcnn/weibo_label2idx.json",
                 max_length=150):
        """
                Args:
                    type: train、inference的选择
                    data_path：使用的数据数据集路径（具体到训练数据位置，用于模型的训练）
                    model：选择模型
                    word2idx_path:textcnn中的word2idx的位置
                    label2idx_path：textcnn中的label2idx位置
                    max_length：最大截断长度

                """
        super().__init__()
        if data_path:
            self.training_data_path = data_path
        if dev_path:
            self.dev_path = dev_path
        if vocab_path:
            self.vocab_set_path = vocab_path
        if word2idx_path:
            self.word2idx_path = word2idx_path
        if label2idx_path:
            self.label2idx_path = label2idx_path
        self.type = type
        self.max_length = max_length
        self.model_list = class_model_list
        # bert模型存储位置
        self.model_path_bert = KNLP_PATH + "/knlp/model/bert/output_model"
        #  beyas模型存储位置
        self.model_path_clf = KNLP_PATH + "/knlp/model/beyas/classification"
        self.model_path_tf = KNLP_PATH + "/knlp/model/beyas/classification"
        # textcnn模型存储位置
        self.model_path_textcnn = KNLP_PATH + "/knlp/model/classification/textcnn.pkl"

    def train(self, model):
        model_list = class_model_list
        if model not in model_list:
            print(f'only support model in {model_list}')
        trainer = ModelTrainer(data_path=self.training_data_pathth,
                            vocab_path=self.vocab_set_path,
                             model=model)
        if model=='beyas':
            trainer.beyas_train(clf_model_path=self.model_path_clf, tf_model_path=self.model_path_tf)
        if model=='bert':
            trainer.bert_train()
        if model=='textcnn':
            trainer.textcnn_train(model_save_path=self.model_path_textcnn,
                                   word2idx_path=self.word2idx_path,
                                   label2idx_path=self.label2idx_path)
        if model=='all':
            trainer.beyas_train(clf_model_path=self.model_path_clf, tf_model_path=self.model_path_tf)
            trainer.bert_train()
            trainer.textcnn_train(model_save_path=self.model_path_textcnn,
                                  word2idx_path=self.word2idx_path,
                                  label2idx_path=self.label2idx_path)

    def inference(self, model, input, model_path_textcnn=None, model_path_bert=None, model_path_clf=None, model_path_tf=None):
        words = input
        model_bert = model_path_bert if model_path_bert else self.model_path_bert
        model_textcnn = model_path_textcnn if model_path_textcnn else self.model_path_textcnn
        model_clf= model_path_clf if model_path_clf else self.model_path_clf
        model_tf = model_path_tf if model_path_tf else self.model_path_tf
        if model not in model_list:
            print(f'only support model in {model_list}')
        else:
            if model == 'bert':
                self.bert_inference(words, model_bert)
            elif model == 'textcnn':
                self.textcnn_inference(words, self.max_length, model_textcnn)
            elif model == 'beyas':
                self.beyas_inference(words, model_clf, model_tf)
            elif model=='all':
                self.bert_inference(words, model_bert)
                self.textcnn_inference(words, self.max_length, model_textcnn)
                self.beyas_inference(words, model_clf, model_tf)

    def bert_inference(self, words, model_path):
        print("******** bert_result ********")
        inference = bertinference('cluener')
        model = BertForTokenClassification.from_pretrained(model_path)
        model.to('cpu')
        result = inference.predict(model=model, text=words)
        print(result)

    def textcnn_inference(self, words, max_length, model_path, word2idx_path, label2idx_path):
        print("******** textcnn_result ********")
        model_path_textcnn = model_path
        tokenizer = jieba.lcut
        inference = InferenceTextCNN(model_path=model_path_textcnn, word2idx_path=word2idx_path,
                                     label2idx_path=label2idx_path, max_length=max_length, tokenizer=tokenizer)
        print(inference([words], return_label=True))

    def beyas_inference(self, words, clf_model, tf_model):
        print("******** beyas_result ********")
        clf_model = clf_model
        tf_model = tf_model
        inference = beyas_inference(clf_model, tf_model)
        MODEL, TF = inference.load_model()
        result = inference.predict(words, MODEL, TF)
        print(result)


if __name__ == '__main__':
    sentence = '我很开心'
    pipe = ClassificationPipeline(data_sign='msra')
    pipe.inference(model='all', input=sentence)
