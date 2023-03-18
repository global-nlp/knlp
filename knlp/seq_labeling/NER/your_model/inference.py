# -*-coding:utf-8-*-
from knlp.seq_labeling.NER.Inference.Inference import NERInference


class YourModelInference(NERInference):
    """
    自定义模型的推理类
    """
    def __init__(self, training_data_path):
        super().__init__()

    def predict(self, input):
        """
        定义模型用于推理的方法，自行增加参数。
        将预测结果按照以下方式存储：
        self.tag_list：预测得到的标签序列
        self.entity_set：预测得到的所以实体的集合
        self.out_sent：按照预测标签进行分割后的句子。
        确保能在pipeline中通过get_tag/get_entity/get_sent获得预测结果。
        """
        pass

