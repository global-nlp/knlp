# !/usr/bin/python
# -*- coding:UTF-8 -*-
from knlp.seq_labeling.crf.inference import Inference
from knlp.seq_labeling.crf.train import Train
from knlp.utils.util import get_model_crf_hanzi_file, get_data_hanzi_segment_file

# init trainer and inferencer
crf_inferencer = Inference()
crf_trainer = Train()


def crf_train(training_data_path, model_save_file):
    """
    This function call crf trainer and inference. You could just prepare training data and test data to build your own
    model from scratch.

    Args:
        training_data_path:

    Returns:

    """
    crf_trainer.init_variable(training_data_path=training_data_path)
    crf_trainer.load_and_train()
    crf_trainer.save_model(model_save_path=model_save_file)
    print(
        "Congratulations! You have completed the training of crf model for yourself. "
        f"Your training info: training_data_path: {training_data_path}. "
        f"model_save_path: {model_save_file}"
    )


def load_and_test_inference(model_save_file, sentence):
    """
    测试推理
    Args:
        model_save_file: string
        sentence: string

    Returns:

    """
    crf_inferencer.spilt_predict(file_path=model_save_file, in_put=sentence)
    print("POS结果：" + str(crf_inferencer.label_prediction))
    print("模型预测结果：" + str(crf_inferencer.out_sentence))


if __name__ == '__main__':
    model_save_file = get_model_crf_hanzi_file()
    crf_train(training_data_path=get_data_hanzi_segment_file(), model_save_file=model_save_file)

    sentence = "从明天起，做一个幸福的人，关心粮食与蔬菜。"
    load_and_test_inference(model_save_file=model_save_file, sentence=sentence)
