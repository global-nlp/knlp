# -*-coding:utf-8-*-
"""
针对CRF模型的类定义
（注：导入pickle用于储存对象化的模型，便于存储用于不同任务的训练后的模型。）

CRFModel类：
由sklearn_crfsuite中的CRF类初始化而来。其中包括训练算法、l1正则与l2正则的系数，迭代次数等模型参数。
"""
import pickle

from knlp.seq_labeling.crf.crf_utils import sentence2features
from sklearn_crfsuite import CRF


class CRFModel(object):
    """
    主要借由sklearn_crfsuite的CRF实现，CRF主要参数列表如下：（所有参数请参考源码）

    官方API文档：https://sklearn-crfsuite.readthedocs.io/en/latest/

    Parameters（参数）
    ----------
    algorithm（算法） : str, optional (default='lbfgs')
        Training algorithm. Allowed values:

        * ``'lbfgs'`` - Gradient descent using the L-BFGS method   使用L-BFGS方法进行梯度下降
        * ``'l2sgd'`` - Stochastic Gradient Descent with L2 regularization term  使用L2正则的随机梯度下降
        * ``'ap'`` - Averaged Perceptron  使用平均感知机（AP）
        * ``'pa'`` - Passive Aggressive (PA)  使用被动攻击算法（PA）
        * ``'arow'`` - Adaptive Regularization Of Weight Vector (AROW)  权重向量自适应正则（AROW），适用于标签中噪声多的情况

    min_freq : float, optional (default=0)
        Cut-off threshold for occurrence
        frequency of a feature. CRFsuite will ignore features whose
        frequencies of occurrences in the training data are no greater
        than `min_freq`. The default is no cut-off.

        设置min_freq之后，对于出现次数低于该参数的特征会进行忽略过滤，有利于控制训练时长。


    all_possible_transitions : bool, optional (default=False)
        Specify whether CRFsuite generates transition features that
        do not even occur in the training data (i.e., negative transition
        features). When True, CRFsuite generates transition features that
        associate all of possible label pairs. Suppose that the number
        of labels in the training data is L, this function will
        generate (L * L) transition features.
        This function is disabled by default.

        若设置为True，CRFsuite会生成在训练数据中未出现过的转移特征，将所有的标签对联系起来，会产生额外的开销，预设值为False。

    c1 : float, optional (default=0)
        The coefficient for L1 regularization.
        If a non-zero value is specified, CRFsuite switches to the
        Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method.
        The default value is zero (no L1 regularization).

        Supported training algorithms: lbfgs

        L1正则化的系数设置。如果不为0，则算法（algorithm）会自动转换到lbfgs的方法继续进行。

    c2 : float, optional (default=1.0)
        The coefficient for L2 regularization.

        Supported training algorithms: l2sgd, lbfgs

        L2正则化的系数设置。支持L2正则的随机梯度下降和lbfgs算法。

    max_iterations : int, optional (default=None)
        The maximum number of iterations for optimization algorithms.
        Default value depends on training algorithm:

        * lbfgs - unlimited;
        * l2sgd - 1000;
        * ap - 100;
        * pa - 100;
        * arow - 100.

        对优化算法的最大迭代数设置。
    """

    def __init__(self):
        self.model = CRF(algorithm='lbfgs',
                         c1=0.1,
                         c2=0.1,
                         max_iterations=300,
                         all_possible_transitions=False,
                         verbose=True)

    def train(self, sentences, tag_lists):
        features = [sentence2features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    def test(self, sentences):
        features = [sentence2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists

    def load_model(self, file_name):
        """用于加载模型"""
        with open(file_name, "rb") as f:
            model = pickle.load(f)

        return model
