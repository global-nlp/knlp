import json
import os
import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, BertTokenizer
import torch.nn.functional as F
from knlp.common.constant import KNLP_PATH
from knlp.seq_labeling.bert.processors.classification import TnewsProcessor as processor
from knlp.seq_labeling.classification.bert.trainer import BertTrain
from knlp.utils.tokenization import BasicTokenizer
from knlp.seq_labeling.bert.tools.progressbar import ProgressBar
from knlp.seq_labeling.bert.tools.collate_fn import collate_fn

BERT_MODEL_PATH = KNLP_PATH + "/knlp/model/bert/output_modelbert"

class bertinference():
    def __init__(self, task):
        self.task = task

    def predict(self, model, text):
        tokenizer = BasicTokenizer(vocab_file=KNLP_PATH + '/knlp/data/msra_bios/vocab.txt', do_lower_case=True)
        nb_pred_steps = 0
        preds = None
        pbar = ProgressBar(n_total=len(text), desc="Predicting")
        input_tokens = tokenizer.tokenize(text)
        for step, batch in enumerate(pred_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': input_tokens[0],
                            'attention_mask': input_tokens[1],
                            'labels': input_tokens[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = model(**inputs)
                _, logits = outputs[:2]
            nb_pred_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            pbar(step)
        print(' ')
        predict_label = np.argmax(preds, axis=1)
        return predict_label

if __name__ == '__main__':
    inference = bertinference('cluener')
    to_be_pred = '我还行'
    model = BertForTokenClassification.from_pretrained(KNLP_PATH + '/knlp/model/bert/output_modelbert/checkpoint-448')
    model.to('cpu')
    processor = processor()
    label_list = processor.get_labels()
    result = inference.predict(model=model, label_list=label_list)
