from knlp.common.constant import KNLP_PATH
from knlp.seq_labeling.NER.pipeline import NERPipeline

if __name__ == '__main__':
    # 获得pipeline对象
    # 这里输入可以参考 knlp/seq_labeling/NER/pipeline.py 中类的参数列表。
    # 值得一提的是data_sign不能为空，需要指定是哪个数据集，因为bert_mrc需要根据data_sign获取对应的实体描述文件。
    pipe = NERPipeline(data_sign='clue', data_path=KNLP_PATH + '/knlp/data/bios_clue/train.bios')
    # 测试训练，输入model指定训练哪一种模型（或全部）。
    pipe.train(model='all')
    # 测试推理，输入语句与指定模型进行推理。
    pipe.inference(input='毕业于北京大学的他，最爱读的书是《时间简史》。', model='all')
    # 测试评估，输入可以是对比两个模型，也可以是单一模型自我评估。
    pipe.eval_interpret('bert_mrc', 'bert_tagger')
