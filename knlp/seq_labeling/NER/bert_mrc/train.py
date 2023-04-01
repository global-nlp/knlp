import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer, AdamW

from knlp.common.constant import KNLP_PATH
from knlp.nn.bilstm_crf.train_nn import TrainNN
from knlp.seq_labeling.bert.data_load import MRCNERDataLoader
from knlp.seq_labeling.bert.models.bert_for_ner import BertQueryNER
from knlp.seq_labeling.bert.tools.common import logger, init_logger
from knlp.utils.metrics.functional.mrc_ner_evaluate import flat_ner_performance, nested_ner_performance


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", default=KNLP_PATH + "/knlp/model/bert/Chinese_wwm/bert_config.json", type=str)
    parser.add_argument("--bert_model", default=KNLP_PATH + "/knlp/model/bert/Chinese_wwm", type=str, )
    parser.add_argument("--output_dir", type=str, default=KNLP_PATH + "/knlp/model/bert/mrc")
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--dev_batch_size", default=4, type=int)
    parser.add_argument("--test_batch_size", default=4, type=int)
    parser.add_argument("--checkpoint", default=1000, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--do_train", default=True, action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--export_model", type=bool, default=True)

    parser.add_argument("--weight_start", type=float, default=1.0)
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument("--weight_span", type=float, default=1.0)
    parser.add_argument("--entity_sign", type=str, default="flat")  # nested flat
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--entity_threshold", type=float, default=0.5)
    parser.add_argument("--data_cache", type=bool, default=True)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--clip_grad", type=int, default=1)
    return parser


class MRCTrain(TrainNN):
    def __init__(self, device: str = "cuda", data_path=None, data_sign=None, vocab_path=None, save_path=None):
        super().__init__(device=device)
        self.training_data_path = KNLP_PATH + '/knlp/data/clue_mrc' if not data_path else data_path
        self.task = data_sign
        self.output_dir = save_path if save_path else KNLP_PATH + '/knlp/model/bert/mrc_ner'
        self.vocab_path = vocab_path if vocab_path else KNLP_PATH + '/knlp/data/bios_clue/vocab.txt'

    def train(self, args, train_dataset, model, tokenizer, label_list):
        param_optimizer = list(model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=10e-8)
        sheduler = None

        dataset_loaders = MRCNERDataLoader(args, label_list, tokenizer, self.vocab_path, mode="train", )

        num_train_steps = dataset_loaders.get_num_train_epochs()
        tr_loss = 0.0
        nb_tr_examples = 0
        nb_tr_steps = 0

        dev_best_acc = 0.0
        dev_best_precision = 0.0
        dev_best_recall = 0.0
        dev_best_f1 = 0.0
        dev_best_loss = float("inf")

        model.train()
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)

        for idx in range(int(args.num_train_epochs)):
            print('\n')
            print("#######" * 10)
            print("EPOCH: ", str(idx))
            for step, batch in tqdm(enumerate(train_dataset)):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, start_pos, end_pos, ner_cate = batch
                loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                             start_positions=start_pos, end_positions=end_pos)
                model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip_grad)
                optimizer.step()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if nb_tr_steps > 0 and nb_tr_steps % args.checkpoint == 0:
                    tmp_dev_loss, tmp_dev_acc, tmp_dev_precision, tmp_dev_recall, tmp_dev_f1 = self.evaluate(idx,
                                                                                                             loss,
                                                                                                             model,
                                                                                                             tokenizer,
                                                                                                             args,
                                                                                                             label_list,
                                                                                                             )

                    if tmp_dev_f1 > dev_best_f1:
                        dev_best_acc = tmp_dev_acc
                        dev_best_loss = tmp_dev_loss
                        dev_best_precision = tmp_dev_precision
                        dev_best_recall = tmp_dev_recall
                        dev_best_f1 = tmp_dev_f1

                        # export model
                        if args.export_model:
                            model_to_save = model.module if hasattr(model, "module") else model
                            output_model_file = os.path.join(self.output_dir,
                                                             "best_checkpoint.bin".format(nb_tr_steps))
                            logger.info("Saving model checkpoint to %s", self.output_dir)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Saving optimizer and scheduler states to %s", self.output_dir)

                        logger.info('\n')
                        logger.info("Best DEV:")
                        logger.info(f"Loss: {dev_best_loss}")
                        logger.info(f"Accuracy: {dev_best_acc}")
                        logger.info(f"Precision: {dev_best_precision}")
                        logger.info(f"Recall: {dev_best_recall}")
                        logger.info(f"F1: {dev_best_f1}")

    def evaluate(self, current_epoch, train_loss, model, tokenizer, args, label_list):
        device = self.device

        dataset_loaders = MRCNERDataLoader(args, label_list, tokenizer, vocab_path=self.vocab_path, mode="train")
        eval_dataset = dataset_loaders.get_dataloader(data_sign="dev")
        model.eval()
        eval_loss = 0

        start_pred_lst = []
        end_pred_lst = []

        mask_lst = []
        start_gold_lst = []
        end_gold_lst = []

        eval_steps = 0
        ner_cate_lst = []

        for input_ids, input_mask, segment_ids, start_pos, end_pos, ner_cate in eval_dataset:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            start_pos = start_pos.to(device)
            end_pos = end_pos.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, start_pos, end_pos)
                start_logits, end_logits = model(input_ids, segment_ids, input_mask)

            start_pos = start_pos.to("cpu").numpy().tolist()
            end_pos = end_pos.to("cpu").numpy().tolist()

            start_label = start_logits.detach().cpu().numpy().tolist()
            end_label = end_logits.detach().cpu().numpy().tolist()

            input_mask = input_mask.to("cpu").detach().numpy().tolist()

            ner_cate_lst += ner_cate.numpy().tolist()
            eval_loss += tmp_eval_loss.mean().item()
            mask_lst += input_mask
            eval_steps += 1

            start_pred_lst += start_label
            end_pred_lst += end_label

            start_gold_lst += start_pos
            end_gold_lst += end_pos

        span_pred_lst = [[[1] * len(start_pred_lst[0])] * len(start_pred_lst[0])] * len(start_pred_lst)
        span_gold_lst = [[[1] * len(start_gold_lst[0])] * len(start_gold_lst[0])] * len(start_gold_lst)

        if args.entity_sign == "flat":
            eval_accuracy, eval_precision, eval_recall, eval_f1, pred_bmes_label_lst = flat_ner_performance(
                start_pred_lst,
                end_pred_lst,
                span_pred_lst,
                start_gold_lst,
                end_gold_lst,
                span_gold_lst,
                ner_cate_lst,
                label_list,
                threshold=args.entity_threshold,
                dims=2)
        else:
            eval_accuracy, eval_precision, eval_recall, eval_f1, pred_bmes_label_lst = nested_ner_performance(
                start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst,
                label_list, threshold=args.entity_threshold, dims=2)

        average_loss = round(eval_loss / eval_steps, 4)
        eval_f1 = round(eval_f1, 4)
        eval_precision = round(eval_precision, 4)
        eval_recall = round(eval_recall, 4)
        eval_accuracy = round(eval_accuracy, 4)

        logger.info("\n")
        logger.info("***** Eval results *****")
        info = "epoch:{} train_loss:{}".format(
            current_epoch, average_loss
        )
        logger.info(info)
        logger.info(f'Average_loss-tep_dev: {average_loss}')
        logger.info(f'Acc-tep_dev: {eval_accuracy}')
        logger.info(f'Precision-tep_dev: {eval_precision}')
        logger.info(f'Recall-tep_dev: {eval_recall}')
        logger.info(f'F1-tep_dev: {eval_f1}')

        return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1

    def run(self):
        args = get_argparse().parse_args()
        args.data_dir = self.training_data_path
        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        os.makedirs(self.output_dir, exist_ok=True)
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        init_logger(log_file=self.output_dir + f'/bert-mrc-{time_}.log')
        if args.do_train:
            model = BertQueryNER(args)
            model.to(self.device)
            with open(os.path.join(self.training_data_path, f'{self.task}.json'), 'r') as fp:
                label_list = json.loads(fp.read())['labels']
            label_list = label_list + ['O']
            tokenizer = BertTokenizer.from_pretrained(args.bert_model)
            dataset_loaders = MRCNERDataLoader(args, label_list, tokenizer, self.vocab_path, mode="train", )
            train_dataset = dataset_loaders.get_dataloader(data_sign="train")

            self.train(args, train_dataset, model, tokenizer, label_list)


if __name__ == '__main__':
    print('Bert-阅读理解训练开始')
    trainer = MRCTrain()
    trainer.run()
    print('Bert-阅读理解训练结束')
