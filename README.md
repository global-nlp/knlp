# knlp

本工具包定位为中文NLP工具包。目标用户分为两类：
1. 无需二次开发：直接使用者
    1. 直接利用工具包提供的模型进行推理
    2. 调用工具包进行模型性能的评估
2. 有二次开发需求：框架利用者
    1. 评估自己训练的模型
    2. 利用自己的数据训练并进行调用

欢迎提出issue或者私信交流

# 安装方式
```
# 直接使用
pip install knlp

# FROM GITHUB SOURCE CODE
pip install git+https://github.com/DukeEnglish/knlp.git

# 本地开发使用
下载好后，在本地目录下使用以下命令：
pip install -e .
这个命令是开发者模式，将会build一个soft link到python包路径下，此时在这个路径下的各种改动可以直接影响到python中安装的包

```
# 示例方法
```python
from knlp import Knlp

def test_all():
    with open("knlp/data/pytest_data.txt") as f:
        text = f.read()
    res = Knlp(text)
    print("seg_result is", res.seg_result)
    print("ner_result is", res.ner_result)
    print("sentiment score is", res.sentiment)
    print("key_words are", res.key_words)
    print("key sentences are", res.key_sentences)
    gt_string = '就读 于 中国人民大学 电视 上 的 电影 节目 项目 的 研究 角色 本人 将 会 参与 配音'
    pred_string = '就读 于 中国 人民 大学 电视 上 的 电影 节目 项 目的 研究 角色 本人 将 会 参与 配音'
    print("evaluation res are", res.evaluation_segment(gt_string, pred_string))
    abs_path_to_gold_file = ''
    abs_path_to_pred_file = ''
    gt_file_name = f'{abs_path_to_gold_file}'
    pred_file_name = f'{abs_path_to_pred_file}'
    print("evaluation file res are", res.evaluation_segment_file(gt_file_name, pred_file_name))
```
其他示例使用方法在samples中。所有的训练数据都在data中有示例数据。

# 测试使用
测试中仅仅包括如何使用现有模型进行inference。由于测试文件中有相对路径存在，所以需要在knlp的项目根目录下运行
```
python test/test_all.py
```
可以看到测试结果。并且可以直接参考其中的代码直接进行inference。

# sample使用方法
这里提供的sample方法会详细包括训练到推理，能够使用本工具训练出一个自己的模型。
- 序列标注的训练
    - 首先生成训练数据，序列标注的数据处理方法在knlp/seq_labeling/data_helper.py。数据针对的是人民日报的数据。
    - 借助knlp使用hmm进行分词训练，生成自己的分词器，并调用自己的分词器：samples/hmm_sample.py，进行hmm的训练：https://zhuanlan.zhihu.com/p/358825066
    - 借助knlp使用crf进行分词训练，存储自己的分词模型：samples/hmm_sample.py，进行crf的训练：https://zhuanlan.zhihu.com/p/489288397

- 信息提取（关键词、关键短语、摘要）
    - samples/IE_sample.py


# 参考并致谢
在实现过程中，调研了网络上很多已经开源的工具包，对他们致以深深的感谢。在coding过程中，参考学习了很多参考pkg中的编码方式，也有直接调用。如果作者感觉到被冒犯，请随时私信联系。

- snownlp
- jieba
- textblob
- sklearn-crfsuite
- https://www.letiantian.me/2014-06-10-pagerank/

# 评估结果
离线评估

CLUE榜单评估结果

# NLP新人入门
推荐阅读：https://dukecourse.feishu.cn/docs/doccnJF2Xt8xHtGf0P9RSHO3eBb
