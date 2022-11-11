import argparse
import glob
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import get_linear_schedule_with_warmup, AdamW
from knlp.common.constant import KNLP_PATH
from transformers import WEIGHTS_NAME, BertConfig
from knlp.nn.bilstm_crf.train_nn import TrainNN
from knlp.seq_labeling.bert.callback.adversarial import FGM
from knlp.seq_labeling.bert.callback.lr_scheduler import get_linear_schedule_with_warmup
from knlp.seq_labeling.bert.callback.optimizater.adamw import AdamW
from knlp.seq_labeling.bert.metrics.ner_metrics import SeqEntityScore
from knlp.seq_labeling.bert.models.bert_for_ner import BertSoftmaxForNer
from knlp.seq_labeling.bert.processors.ner_seq import collate_fn
from knlp.seq_labeling.bert.processors.ner_seq import convert_examples_to_features
from knlp.seq_labeling.bert.tools.common import logger, init_logger
from knlp.seq_labeling.bert.tools.common import seed_everything
from knlp.utils.tokenization import BasicTokenizer
from knlp.seq_labeling.bert.processors.ner_seq import SelectProcessor as processors


def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir", default=KNLP_PATH + "/knlp/data/cluener_public", type=str, required=False,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
    parser.add_argument("--model_type", default='bert', type=str, required=False,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default=KNLP_PATH + "/knlp/model/bert/Chinese_wwm", type=str,
                        required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default=KNLP_PATH + "/knlp/model/bert/output_model", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.", )

    # Other parameters
    parser.add_argument('--markup', default='bios', type=str,
                        choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", default=True, action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", default=True, action="store_true",
                        help="Set this flag if you are using an uncased model.")
    # adversarial training
    parser.add_argument("--do_adv", action="store_true",
                        help="Whether to adversarial training.")
    parser.add_argument('--adv_epsilon', default=1.0, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")

    parser.add_argument("--per_gpu_train_batch_size", default=24, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=24, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=int, default=200,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", default=False, action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", default=KNLP_PATH + "/knlp/model/bert/output_model",
                        action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    return parser


class BERTTrain(TrainNN):
    def __init__(self, device: str = "cuda", data_path=None, tokenizer_vocab=None, data_sign=None, save_path=None):

        super().__init__(device=device)
        self.output_dir = save_path if save_path else KNLP_PATH + "/knlp/model/bert/output_model"
        self.vocab = KNLP_PATH + '/knlp/model/bert/Chinese_wwm/vocab.txt' if not tokenizer_vocab else tokenizer_vocab
        self.tokenizer = BasicTokenizer(vocab_file=self.vocab,
                                        do_lower_case=True)
        self.training_data_path = data_path
        self.task = data_sign

    def train(self, args, train_dataset, model, tokenizer):
        """ Train the model """
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=collate_fn)
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        args.warmup_steps = int(t_total * args.warmup_proportion)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        global_step = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        if args.do_adv:
            fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)
        model.zero_grad()
        seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
        # pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
        if args.save_steps == -1 and args.logging_steps == -1:
            args.logging_steps = len(train_dataloader)
            args.save_steps = len(train_dataloader)
        for epoch in range(int(args.num_train_epochs)):
            # pbar.reset()
            # pbar.epoch_start(current_epoch=epoch)
            for step, batch in enumerate(train_dataloader):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if args.do_adv:
                    fgm.attack()
                    loss_adv = model(**inputs)[0]
                    if args.n_gpu > 1:
                        loss_adv = loss_adv.mean()
                    loss_adv.backward()
                    fgm.restore()
                # pbar(step, {'loss': loss.item()})
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        print(" ")
                        self.evaluate(args, model, tokenizer)

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(self.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = (model.module if hasattr(model, "module") else model)
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        # tokenizer.save_vocabulary(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)
            logger.info("\n")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
        return global_step, tr_loss / global_step

    def evaluate(self, args, model, tokenizer, prefix=""):
        metric = SeqEntityScore(args.id2label, markup=args.markup)
        eval_output_dir = self.output_dir
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)
        eval_dataset = self.load_and_cache_examples(args, self.task, tokenizer, data_type='dev')
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_fn)
        # Eval!
        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        # pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
            out_label_ids = inputs['labels'].cpu().numpy().tolist()
            input_lens = batch[4].cpu().numpy().tolist()
            for i, label in enumerate(out_label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif j == input_lens[i] - 1:
                        metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                        break
                    else:
                        temp_1.append(args.id2label[out_label_ids[i][j]])
                        temp_2.append(preds[i][j])
            # pbar(step)
        logger.info("\n")
        eval_loss = eval_loss / nb_eval_steps
        eval_info, entity_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss
        logger.info("***** Eval results %s *****", prefix)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        logger.info(info)
        logger.info("***** Entity results %s *****", prefix)
        for key in sorted(entity_info.keys()):
            logger.info("******* %s results ********" % key)
            info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
            logger.info(info)
        return results

    def load_and_cache_examples(self, args, task, tokenizer, data_type='train'):
        # if args.local_rank not in [-1, 0] and not args.do_eval:
        #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        # processor = processors[task]()
        processor = processors(self.task)
        if self.training_data_path:
            args.data_dir = self.training_data_path
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(args.data_dir, 'cached_soft-{}_{}_{}_{}'.format(
            data_type,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
            str(task)))
        if os.path.exists(cached_features_file) and not args.overwrite_cache and 1 == 0:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            if data_type == 'train':
                examples = processor.get_train_examples(args.data_dir)
            elif data_type == 'dev':
                examples = processor.get_dev_examples(args.data_dir)
            else:
                examples = processor.get_test_examples(args.data_dir)
            features = convert_examples_to_features(examples=examples,
                                                    tokenizer=tokenizer,
                                                    label_list=label_list,
                                                    max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                        else args.eval_max_seq_length,
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=0,
                                                    sep_token=tokenizer.sep_token,
                                                    pad_token_segment_id=0,
                                                    )
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
        return dataset

    def run(self):
        args = get_argparse().parse_args()
        if self.training_data_path:
            args.data_dir = self.training_data_path
        tokenizer = self.tokenizer
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.output_dir = self.output_dir + '{}'.format(args.model_type)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        init_logger(log_file=self.output_dir + f'/{args.model_type}-{self.task}-{time_}.log')
        if os.path.exists(self.output_dir) and os.listdir(
                self.output_dir) and args.do_train and not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    self.output_dir))
        if  args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            args.n_gpu = 1
        args.device = self.device
        # Set seed
        seed_everything(args.seed)
        # Prepare NER task
        self.task = self.task.lower()

        # processor = processors[self.task]()
        processor = processors(self.task)
        label_list = processor.get_labels()
        args.id2label = {i: label for i, label in enumerate(label_list)}
        args.label2id = {label: i for i, label in enumerate(label_list)}
        num_labels = len(label_list)

        # # Load pretrained model and tokenizer
        # if args.local_rank not in [-1, 0]:
        #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        args.model_type = args.model_type.lower()
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.loss_type = args.loss_type
        model = BertSoftmaxForNer.from_pretrained(args.model_name_or_path, config=config)
        # if args.local_rank == 0:
        #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)
        logger.info("Training/evaluation parameters %s", args)
        # Training
        if args.do_train:
            train_dataset = self.load_and_cache_examples(args, self.task, tokenizer, data_type='train')
            global_step, tr_loss = self.train(args, train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
            # Create output directory if needed
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            logger.info("Saving model checkpoint to %s", self.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(self.output_dir)
            # tokenizer.save_vocabulary(self.output_dir)
            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(self.output_dir, "training_args.bin"))

        # Evaluation
        results = {}
        if args.do_eval:
            checkpoints = [self.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in
                    sorted(glob.glob(self.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
                model = BertSoftmaxForNer.from_pretrained(checkpoint)
                model.to(args.device)
                result = self.evaluate(args, model, tokenizer, prefix=prefix)
                if global_step:
                    result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
                results.update(result)
            output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))


if __name__ == '__main__':
    print('Bert-序列标注训练开始')
    trainer = BERTTrain()
    trainer.run()
    print('Bert-序列标注训练结束')
