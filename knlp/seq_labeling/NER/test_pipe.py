from pipeline import Pipeline

if __name__ == '__main__':
    pipe = Pipeline(data_sign='clue')
    # pipe.train(model='crf')
    pipe.inference(input='毕业于北京大学的他，最爱读的书是《时间简史》。', model='crf')
    # pipe.train(model='bilstm')
    # pipe.eval_interpret('bert_mrc', 'bert_tagger')
