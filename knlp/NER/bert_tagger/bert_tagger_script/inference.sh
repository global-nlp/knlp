#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: inference.sh


REPO_PATH=/Users/ericmac/knlp/knlp
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=zh_msra
DATA_DIR=/Users/ericmac/knlp/knlp/data/zh_msra
BERT_DIR=/Users/ericmac/knlp/knlp/model/bert/Chinese_wwm
MAX_LEN=200
OUTPUT_DIR=${REPO_PATH}/model/bert/0628/msra_bert_tagger_chinese_lr2e-5_drop_norm1.0_weight_warmup_maxlen256
MODEL_CKPT=${OUTPUT_DIR}/epoch=9_v1.ckpt
HPARAMS_FILE=${OUTPUT_DIR}/lightning_logs/version_0/hparams.yaml
DATA_SUFFIX=.char.bmes

python3 ${REPO_PATH}/NER/bert_tagger/tagger_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--dataset_sign ${DATA_SIGN} \
--data_file_suffix ${DATA_SUFFIX}