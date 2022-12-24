from pipeline import Pipeline

if __name__ == '__main__':
    pipe = Pipeline(data_sign='msra')
    # pipe.train(model='crf')
    pipe.inference(input='爱看《星际迷航》的王明，毕业于清华大学计算机学院，曾任月球绿色空间站站长。', model='hmm')
    # pipe.train(model='bilstm')
    # pipe.eval_interpret('bilstm', 'crf')
