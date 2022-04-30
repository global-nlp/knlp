## CI（Continue Integrate）自动化持续集成

简单理解CI就是对提交的git代码进行自动化的单元测试、代码风格检查、版本发布等，在多人开发时可以很好地进行规范约束。

### 常见工具

**Github Actioin**：

提供了 Linux、Windows、和 macOS 虚拟机运行你的工作流，在进行push或者pull request时进行工作流中的任务

**Jenkins**：

支持Windows、Mac OSX以及各类Unix系统，可以使用本机系统软件包以及Docker进行安装，也可以在安装了Java Runtime Environment（JRE）的任何机器上独立安装。

 Jenkins提供超过1000款插件选项，可以集成几乎所有市场上可用的工具和服务。

**Travis**：

支持Linux Ubuntu和OSX，免费版有次数限制，需要绑定信用卡，付费版最低69$/月。

**GitLab CI**：

适用于使用GitLab作为项目管理平台的



### Github Actioin 集成

在工程目录下添加yml文件即可

```
.github/workflows/knlp.yml
```

```
name: KNLP
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.7
      uses: actions/setup-python@v3
      with:
        python-version: "3.7"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Lint with flake8
      run: |
        flake8 . --count
    - name: Setup
      run: |
        sudo python setup.py install
    - name: run test_all
      run: |
        sudo python test/test_all.py
```

### Github Actioin yml详解

格式要求：缩进严格规范两个空格

**1.工作流触发条件 on**

push和提交pr的时候触发

```
on:[push、pull_request]
```

指定分支触发

```python
on:
  push:
    branches:
      - introduce_CI
  pull_request:
    branches:
      - main
```

**2.CI需执行的job，build、run**

```
jobs:
  build:
  run:
    needs:build #build 成功再run
```

**3.运行环境**

可选 windows-2019, windows-latest, macos-10.15, macos-latest

```
jobs:
  job1:
    runs-on: ubuntu-latest
  job2:
    runs-on: macos-10.15
```

**4.job步骤 steps**

uses：指定CI官方的版本 https://github.com/actions/checkout

```
- uses: actions/checkout@v3
```

name: 步骤的名字，会在Github Actioin 的job界面中展示

可以根据自己的需要通过脚本来写自己的步骤，简单的操作参考上面的yml配置中的step