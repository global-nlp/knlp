from knlp.common.constant import KNLP_PATH, model_list
from knlp.seq_labeling.NER.pipeline import NERPipeline
from knlp.seq_labeling.NER.preprocess import preprocess_trie, VOCABProcessor, DATAProcessor

if __name__ == '__main__':
    # 进行数据准备
    # trie树准备，需提供训练语料路径
    data_path = KNLP_PATH + '/knlp/data/msra_bios/train.bios'
    mid_path = KNLP_PATH + '/knlp/data/NER_data/dict.txt'
    out_path = KNLP_PATH + '/knlp/data/NER_data/ner_dict.txt'
    state_path = KNLP_PATH + '/knlp/data/NER_data/state_dict.json'
    preprocess_trie(data_path, out_path, state_path)
    # 准备vocab，需提供语料数据文件夹路径
    vocabprocessor = VOCABProcessor(KNLP_PATH + '/knlp/data/msra_bios/')
    vocabprocessor.gen_dict()
    vocabprocessor.merge_vocab()
    vocabprocessor.add_vocab()
    # mrc准备，需提供实体描述json文件
    description_json = KNLP_PATH + '/knlp/data/msra_mrc/msra.json'
    msraProcessor = DATAProcessor(description_json, KNLP_PATH + '/knlp/data/msra_bios/vocab.txt')
    # 第一步：先生成中间文件和标签
    msraProcessor.get_mid_data(KNLP_PATH + '/knlp/data/msra_bios/train.bios',
                               KNLP_PATH + '/knlp/data/msra_bios/train.mid')
    msraProcessor.get_mid_data(KNLP_PATH + '/knlp/data/msra_bios/val.bios',
                               KNLP_PATH + '/knlp/data/msra_bios/dev.mid')
    msraProcessor.get_mid_data(KNLP_PATH + '/knlp/data/msra_bios/test.bios',
                               KNLP_PATH + '/knlp/data/msra_bios/test.mid')
    # 第二步：生成MRC所需要的数据
    msraProcessor.get_mrc_data(KNLP_PATH + '/knlp/data/msra_bios/train.mid',
                               KNLP_PATH + '/knlp/data/msra_mrc/train.mrc')
    msraProcessor.get_mrc_data(KNLP_PATH + '/knlp/data/msra_bios/dev.mid',
                               KNLP_PATH + '/knlp/data/msra_mrc/dev.mrc')
    msraProcessor.get_mrc_data(KNLP_PATH + '/knlp/data/msra_bios/test.mid',
                               KNLP_PATH + '/knlp/data/msra_mrc/test.mrc')

    # 获得pipeline对象
    # 这里输入可以参考 knlp/seq_labeling/NER/pipeline.py 中类的参数列表。
    # 值得一提的是data_sign不能为空，需要指定是哪个数据集，因为bert_mrc需要根据data_sign获取对应的实体描述文件。
    pipe = NERPipeline(data_sign='clue', data_path=KNLP_PATH + '/knlp/data/bios_clue/train.bios')
    # 测试训练，输入model指定训练哪一种模型（或全部）。
    pipe.train(model='all')
    # 测试推理，输入语句与指定模型进行推理。
    pipe.inference(input='毕业于北京大学的他，最爱读的书是《时间简史》。', model='all')
    # 测试评估，输入可以是对比两个模型，也可以是单一模型自我评估。
    for model in model_list:
        # 对单一模型评估
        pipe.eval_interpret(model)
        for __model in model_list:
            # 两两评估
            pipe.eval_interpret(model, __model)
