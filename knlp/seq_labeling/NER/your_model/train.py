"""
模型的训练部分，如果是神经网络模型，建议继承TrainSeqLabel；如果是其它类型如统计学方法，则自行创建类，进行数据加载与训练存储，可以参考crf/train.py。
"""
from knlp.nn.bilstm_crf.train_seq_labeling import TrainSeqLabel


class YourModelTrain(TrainSeqLabel):
    """
    神经网络模型，可以参考以下架构填写方法。值得注意的是，TrainSeqLabel本身带有train方法了，但如果与自定义模型调用方式差异较大，建议重写train方法。
    """

    def __init__(self):
        super().__init__()
        self.model = ...

    def save(self):
        """
        保存模型文件、配置文件
        """
        pass