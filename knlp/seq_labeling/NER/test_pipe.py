from knlp.common.constant import KNLP_PATH
from pipeline import Pipeline

if __name__ == '__main__':
    pipe = Pipeline(data_sign='msra', data_path=KNLP_PATH + '/knlp/data/msra_bios/train.bios')
    pipe.train(model='hmm')
    pipe.inference(input='毕业于北京大学的他，最爱读的书是《时间简史》。', model='hmm')
    # pipe.train(model='bilstm')
    # pipe.eval_interpret('bert_mrc', 'bert_tagger')
