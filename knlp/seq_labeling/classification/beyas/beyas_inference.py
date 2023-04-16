import jieba
import joblib


class beyas_inference():

    def __init__(self, model_path, tf_path):
        self.model_path = model_path
        self.tf_path = tf_path

    def load_model(self):
        global MODEL
        global TF
        MODEL = joblib.load(self.model_path)
        TF = joblib.load(self.tf_path)
        return MODEL, TF

    def predict(sentence, MODEL, TF):
        assert MODEL != None and TF != None
        words = jieba.cut(sentence)
        s = ' '.join(words)
        test_features = TF.transform([s])
        predicted_labels = MODEL.predict(test_features)
        return predicted_labels[0]
