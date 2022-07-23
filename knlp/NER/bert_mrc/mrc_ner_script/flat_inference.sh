#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: flat_inference.sh
#

REPO_PATH=/Users/ericmac/knlp/knlp
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=conll03
DATA_DIR=/Users/ericmac/knlp/knlp/data/zh_msra
BERT_DIR=/Users/ericmac/knlp/knlp/model/bert/Chinese_wwm
MAX_LEN=180
MODEL_CKPT=${REPO_PATH}/model/bert/mrc_bert_out/large_lr3e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen180/epoch=1_v7.ckpt
HPARAMS_FILE=${REPO_PATH}/model/bert/mrc_bert_out/large_lr3e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen180/lightning_logs/version_0/hparams.yaml


python3 ${REPO_PATH}/NER/bert_mrc/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--flat_ner \
--dataset_sign ${DATA_SIGN}