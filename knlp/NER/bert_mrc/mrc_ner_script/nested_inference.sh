#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: nested_inference.sh
#

REPO_PATH=/Users/ericmac/knlp/knlp
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=msra
DATA_DIR=/Users/ericmac/knlp/knlp/data/zh_msra
BERT_DIR=/Users/ericmac/knlp/knlp/model/bert/Chinese_wwm
MAX_LEN=128
MODEL_CKPT=${REPO_PATH}/model/bert/mrc_bert_out/zh_msra_out8e-620200913_dropout0.2_maxlen128/epoch=19.ckpt
HPARAMS_FILE=${REPO_PATH}/model/bert/mrc_bert_out/zh_msra_out8e-620200913_dropout0.2_maxlen128/lightning_logs/version_0/hparams.yaml


python3 ${REPO_PATH}/NER/bert_mrc/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--dataset_sign ${DATA_SIGN}