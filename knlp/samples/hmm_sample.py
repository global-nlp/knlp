# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: hmm_samples
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-03-27
# Description: 实现了使用hmm进行训练并将模型保存在目录中的功能
# -----------------------------------------------------------------------#
from knlp.common.constant import KNLP_PATH
from knlp.seq_labeling.hmm.inference import Inference
from knlp.seq_labeling.hmm.train import Train
from knlp.utils.util import get_pku_vocab_train_file, get_pku_hmm_train_file

# init trainer and inferencer
hmm_inferencer = Inference()
hmm_trainer = Train()


def hmm_train(vocab_set_path, training_data_path, model_save_path):
    """
    This function call hmm trainer and inference. You could just prepare training data and test data to build your own
    model from scratch.

    Args:
        vocab_set_path:
        training_data_path:
        model_save_path:

    Returns:

    """
    hmm_trainer.init_variable(vocab_set_path=vocab_set_path, training_data_path=training_data_path)
    hmm_trainer.build_model(state_set_save_path=model_save_path, transition_pro_save_path=model_save_path,
                            emission_pro_save_path=model_save_path,
                            init_state_set_save_path=model_save_path)
    print(
        "Congratulations! You have completed the training of hmm model for yourself. "
        f"Your training info: vocab_set_path: {vocab_set_path}, training_data_path: {training_data_path}. "
        f"model_save_path: {model_save_path}"
    )


def hmm_inference_load_model(model_save_path):
    """
    load model
    Args:
        model_save_path: string

    Returns:

    """
    hmm_inferencer.load_mode(state_set_save_path=model_save_path, transition_pro_save_path=model_save_path,
                             emission_pro_save_path=model_save_path,
                             init_state_set_save_path=model_save_path)


def test_inference(sentence):
    """
    测试推理
    Args:
        sentence: string

    Returns:

    """
    return list(hmm_inferencer.cut(sentence))


if __name__ == '__main__':
    model_save_path = KNLP_PATH + "/knlp/model/hmm/"
    hmm_train(vocab_set_path=get_pku_vocab_train_file(), training_data_path=get_pku_hmm_train_file(), model_save_path=model_save_path)
    hmm_inference_load_model(model_save_path=model_save_path)
    print(test_inference("大家好，我是你们的好朋友"))
