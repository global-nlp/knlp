from pipeline import Pipeline

if __name__ == '__main__':
    Pipeline(type='inference', input='爱看《星际迷航》的王明，毕业于清华大学计算机学院，曾任月球绿色空间站站长。', model='all', do_eval=False, data_sign='msra')
