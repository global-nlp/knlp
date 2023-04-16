from knlp.common.constant import KNLP_PATH
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


class beyas_train():
    def __init__(self, file_path, clf_model_path, tf_model_path):
        self.file_path = file_path
        self.clf_model_path = clf_model_path
        self.tf_model_path = tf_model_path

    def load_data(self):
        with open(slef.file_path) as f:
            lines = f.readlines()
        data = []
        label = []
        for line in lines:
            line = eval(line)
            words = jieba.cut(line['query'])
            print(words)
            s = ''
            for w in words:
                s += w + ' '
            s = s.strip()
            data.append(s)
            label.append(line['label'])
        return data, label

    def train(self, datas, labels):
        tf = TfidfVectorizer(max_df=0.5)
        train_features = tf.fit_transform(datas)
        clf = MultinomialNB(alpha=0.001).fit(train_features, labels)
        joblib.dump(clf, self.clf_model_path)
        joblib.dump(tf, self.tf_model_path)


if __name__ == '__main__':
    test = beyas_train(KNLP_PATH + '/knlp/seq_labeling/classification/bert/dataset/data_train.json')
    train_datas, train_labels = test.load_data()
    tf = TfidfVectorizer(max_df=0.5)
    train_features = tf.fit_transform(train_datas)
    clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
    joblib.dump(clf, KNLP_PATH + '/knlp/seq_labeling/classification/beyas/model/nb.pkl')
    joblib.dump(tf, KNLP_PATH + '/knlp/seq_labeling/classification/beyas/model/tf.pkl')
