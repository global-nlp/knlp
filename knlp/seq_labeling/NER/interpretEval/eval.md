## The Evaluation Of NER

在对于NER的结果进行评估时，我们使用了论文 “**Interpretable Multi-dataset Evaluation for Named Entity Recognition**” 的方法进行创新性地评估。

### 1 传统评估指标

传统模式中，我们通常会使用F1参数来对NER的结果进行评估，这种方式适合模型之间的快速比较与评估。但它们是不透明的，通过指标，你不会知道模型的优缺点是什么，或者数据的哪些特征对模型的影响最大。

你只能猜测产生误差的原因是否是句子过长、实体太多（或太少）、实体分布得太远（或太近）等。



### 2 创新性评估

卡内基梅隆大学和复旦大学的研究员共同提出了一种新颖的评估技术。主要思想是根据实体长度，标签一致性，实体密度，句子长度等属性将数据划分为多个实体桶，然后分别在每个这些桶上评估模型。

#### 2.1 属性定义

以NER为例，论文共为NER任务设置了8个属性

|  Id  |                       NER                       |
| :--: | :---------------------------------------------: |
|  1   |            实体长度（Entity Length）            |
|  2   |           句子长度（Sentence Length）           |
|  3   |             OOV密度（OOV Density）              |
|  4   |          token频率（Token Frequency）           |
|  5   |          实体频率（Entity Frequency）           |
|  6   | token的标签一致性（Label Consistency of Token） |
|  7   | 实体的标签一致性（Label Consistency of Entity） |

#### 2.2 分桶

分桶将模型的整体性能分解为不同类别。

可以通过将测试实体集划分为测试实体（关于 span 和句子级属性）或测试标记（关于标记级属性）的不同子集来实现。

#### 2.3 细分

计算每个桶的性能

#### 概要统计

使用统计度量总结可量化的结果



### 3 应用

#### 3.1 系统诊断

- 自我诊断
- 辅助诊断

#### 3.2 数据集偏差分析

#### 3.3 结构偏差分析



### 4 结果解释

#### 4.1 网站运行

将文件上传到 [ExplainaBoard](http://explainaboard.nlpedia.ai/) 网站（http://explainaboard.nlpedia.ai/）

#### 4.2 本地运行

以NER任务为例。运行shell：`./run_task_ner.sh`

shell 脚本包括以下三个方面：

- `tensorEvaluation-ner.py`：计算细粒度分析的相关结果。
- `genFig.py`：绘图以显示细粒度分析的结果。
- `genHtml.py`：将上一步中的图形绘制到网页中。

运行上述命令后，将生成一个网页，用于显示模型的分析和诊断结果：`tEval-ner.html`

中文分词任务的运行过程类似。

##### 4.2.1 要求

- `python3`
- `texlive`
- `poppler`
- `pip3 install -r requirements.txt`

##### 4.2.2 对模型进行分析诊断

以CoNLL-2003数据集为例。

- 将模型的结果文件放在此路径上：（它包含三列，由空格分隔：token、true-tag和predicted-tag）。为了执行模型诊断，必须包含两个或多个模型结果文件。您也可以选择我们提供的结果文件之一作为参考模型。`data/ner/conll03/results/`
- 将训练集和测试集（与结果文件相关的数据集）命名为“train.txt”和“test.txt”，然后将它们放在以下路径上：`data/ner/conll03/data/`
- 根据数据设置（训练集的路径）、（数据集名称）、（第一个模型的名称）、（第二个模型的名称）、（结果的路径）。`path_data` `datasets[-]` `model1` `model2` `resfiles[-]` `run_task_ner.sh`
- run：分析结果将在以下路径上生成。`./run_task_ner.sh` `output_tensorEval/ner/your_model_name/`

```
Notably, so far, our system only supports limited tasks and datasets, 
we're extending them currently!
```

##### 4.2.3 生成HTML代码

如 4.2.2 节中所述，我们在路径上生成了分析结果。接下来，我们将在分析结果上生成HTML代码库。在 中，后面的代码用于生成 HTML 代码。在运行之前，您需要确保已安装以下环境 ：`output_tensorEval/ner/your_model_name/` `./run_task_ner.sh` `#run pdflatex .tex` `./run_task_ner.sh` `texlive` `poppler`

该代码的其他说明如下：`./run_task_ner.sh`

- `genFig.py`：生成有关分析图表的代码（例如条形图、热图）`latex`
- `pdflatex $file.tex`：生成基于latex代码的格式的图`.pdf`
- `pdftoppm -png $file.pdf`：将 图形 with 转换为pdf或png格式`.pdf` `.png`
- `genHtml.py`：生成排列分析图和表的 HTML 代码。

##### 4.2.4 注意

- **需要两个以上的结果文件。**由于比较诊断是比较模型体系结构的优缺点以及两个或多个模型之间的预训练知识，因此有必要输入至少两个模型结果。
- **结果文件必须包含一下三列：单词、true 标记和预测标记，以空格分隔。**如果结果文件不是所需的格式，则可以修改文件中的函数以适合您的格式。`read_data()` `tensorEvaluation-ner.py`