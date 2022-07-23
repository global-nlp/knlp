#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: msra.sh

REPO_PATH=/Users/ericmac/knlp/knlp
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
export TOKENIZERS_PARALLELISM=false

DATA_DIR=/Users/ericmac/knlp/knlp/data/zh_msra
BERT_DIR=/Users/ericmac/knlp/knlp/model/bert/Chinese_wwm
SPAN_WEIGHT=0.1
DROPOUT=0.2
LR=8e-6
MAXLEN=128
INTER_HIDDEN=1536

BATCH_SIZE=4
PREC=16
VAL_CKPT=0.25
ACC_GRAD=1
MAX_EPOCH=20
SPAN_CANDI=pred_and_gold
PROGRESS_BAR=1

OUTPUT_DIR=${REPO_PATH}/model/bert/mrc_bert_out/${LR}20200913_dropout${DROPOUT}_maxlen${MAXLEN}

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 python ${REPO_PATH}/NER/bert_mrc/mrc_ner_trainer.py \
--gpus="1" \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAXLEN} \
--batch_size ${BATCH_SIZE} \
--precision=${PREC} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--lr ${LR} \
--val_check_interval ${VAL_CKPT} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--weight_span ${SPAN_WEIGHT} \
--span_loss_candidates ${SPAN_CANDI} \
--chinese \
--workers 0 \
--classifier_intermediate_hidden_size ${INTER_HIDDEN}

