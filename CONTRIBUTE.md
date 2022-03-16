# KNLP开源贡献指南

欢迎您对KNLP的相关工作感兴趣。本文档作为基本指南来指引您如何向KNLP进行贡献。如果您发现文档中有错误或者又缺失的内容，请及时与我们联系



我们随时都欢迎任何贡献，无论是简单的错别字修正，BUG 修复还是增加新功能。请踊跃提出问题或发起 PR。我们同样重视文档以及与其它开源项目的整合，欢迎在这方面做出贡献。

## 从哪里入手？

如果您是初次贡献，可以先从issue或者我们的小任务中开始快速参与社区贡献。

您可以直接在相应 issue 中回复参与意愿，或者提出您想要做的工作，参照下面的 GitHub 工作流指引解决 issue 并按照规范提交 PR，通过 review 后就会被 merge 到 master 分支。



## 如何贡献

开源贡献者常用的工作流（workflow）：

1.将仓库 fork 到自己的 GitHub 下

2.将 fork 后的仓库 clone 到本地，在main分支上创建新分支，进行开发

3.完成开发后，在本地提交变更，**注意：**

**1) commit log 保持简练、规范。 2) 提交的 email 需要和 GitHub 的 email 保持一致 3）通常情况下，请确保对应的变更都有测试用例或 demo 进行验证**

4.将开发分支push本地代码到自己的github上(提交分支前确保证拉取了最新代码)，在github上提交Merge Pull Request



环境**tips：**

1.如果运行测试用例，import相关包存在冲突，可在setup.py路径下运行 pip install -e .

2.如果存在import 该工程下的相关模块 路径报错，可检查project structure 中 项目路径设置是否为knlp项目所在路径(pycharm 默认会加载python site-package 的路径)



## 创建 Issue / PR

我们使用 GitHub Issues 以及 Pull Requests 来管理/追踪问题。

如果您发现了文档中有表述错误，或者代码发现了 BUG，或者希望开发新的特性，或者希望提建议，可以创建一个 Issue。请参考 Issue 模板中对应的指导信息来完善 Issue 的内容，来帮助我们更好地理解您的 Issue。

如果您想要贡献代码，您可以参考上面的 [GitHub 工作流]，提交对应的 PR。若是对当前开发版本进行提交，则目标分支为 `master`。如果您的 PR 包含非常大的变更，比如模块的重构或者添加新的组件，请**务必先提出相关 issue，发起详细讨论，达成一致后再进行变更**，并为其编写详细的文档来阐述其设计、解决的问题和用途。注意一个 PR 尽量不要过于大。如果的确需要有大的变更，可以将其按功能拆分成多个单独的 PR。

## 报告安全问题

特别地，若您发现 CLUE 及其生态项目中有任何的安全漏洞（或潜在的安全问题），请第一时间通过邮箱[[chineseGLUE@163.com](mailto:chineseGLUE@163.com)私下联系我们。在对应代码修复之前，**请不要将对应安全问题对外披露，也不鼓励公开提 issue 报告安全问题**。



## Contribution guidelines and standards

在你提交你的PR之前，麻烦请确定你的改变符合我们的规范

#### General guidelines and philosophy for contribution

- 如果你贡献了一个新的特性，请尽量包含你的单元测试以保证你的代码可以使用，并且降低未来的维护成本。
- 修复了bug也需要写单元测试
- 请维持API的兼容性

### Code review

所有的代码都需要经过 committer 进行 review。以下是我们推荐的一些原则：

- 可读性：代码遵循我们的开发规约，重要代码需要有详细注释和文档
- 优雅性：代码简练、复用度高，有着完善的设计
- 测试：重要的代码需要有完善的测试用例（单元测试、集成测试），对应的衡量标准是测试覆盖率

#### Python coding style

Use `pylint` to check your Python changes. To install `pylint` and check a file with `pylint` against TensorFlow's custom style definition:

We encourage PEP-8.

```
pip install pylint
pylint myfile.py
```

Note `pylint `should run from the top level directory.

#### Running unit tests

We encourage you to send your PR with your test case. Then, the review process will be quick.

# 社区

## 联系我们

### 邮件组

如果您有任何问题与建议，请通过邮箱[chineseGLUE@163.com](mailto:chineseGLUE@163.com)联系我们。

### Gitter

我们的 Gitter room: https://github.com/CLUEbenchmark

以上贡献者模版参考自：Sentinel and [Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)。感谢他们的智慧。