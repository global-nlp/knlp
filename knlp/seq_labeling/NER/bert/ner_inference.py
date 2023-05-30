import json
import os

import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, BertTokenizer
import torch.nn.functional as F

from knlp.common.constant import KNLP_PATH
from knlp.seq_labeling.NER.Inference.Inference import NERInference
from knlp.seq_labeling.bert.processors.ner_seq import select_processor as processors, logger, collate_fn, InputFeatures
from knlp.seq_labeling.bert.models.bert_for_ner import BertSoftmaxForNer
from knlp.utils.get_entity import get_entities
from knlp.utils.tokenization import BasicTokenizer

BERT_MODEL_PATH = KNLP_PATH + "/knlp/model/bert/output_modelbert"

texts = [
    ('1945年', 1, None),
    ('8月', 1, None),
    ('爱琴海', 1, 'sce')
]


class BertInference(NERInference):
    def __init__(self, task, model_name=BERT_MODEL_PATH, log=False):
        super().__init__()
        self.task = task
        self.model = model_name
        self.token = []
        if log:
            self.detailed_log()

    def token_predict(self):
        return self.token

    def predict(self, text, model, vocab_path, mask_padding_with_zero=True, max_seq_length=512, sep_token="[SEP]",
                sequence_a_segment_id=0, cls_token="[CLS]", cls_token_segment_id=1, pad_token=0,
                pad_token_segment_id=0):
        features = []
        basicTokenizer = BasicTokenizer(vocab_file=vocab_path, do_lower_case=True)
        input_tokens = basicTokenizer.tokenize(text)

        special_tokens_count = 2
        if len(input_tokens) > max_seq_length - special_tokens_count:
            input_tokens = input_tokens[: (max_seq_length - special_tokens_count)]
        input_tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(input_tokens)
        input_tokens = [cls_token] + input_tokens
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = basicTokenizer.convert_tokens_to_ids(tokens=input_tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)

        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        input_len = len(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                      segment_ids=segment_ids, label_ids=None))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)

        inputs = {"input_ids": all_input_ids, "attention_mask": all_input_mask, "labels": None}

        output = model(**inputs)

        logits = output[0]
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        preds = preds[0][1:-1]
        processor = processors(self.task)
        label_list = processor.get_labels()
        id2label = {i: label for i, label in enumerate(label_list)}
        tags = [id2label[x] for x in preds]

        sentence = input_tokens[1:-1]
        tag_list = tags[:len(input_tokens) - 2]

        self.cut_bio(sentence, tag_list)

        self.tag_list = tag_list
        self.token = sentence

        if self.log:
            print("tags: " + str(tags[:len(input_tokens) - 2]))
            print("tokens: " + str(input_tokens[1:-1]))
            print("input_ids: " + str(input_ids))
            print("input_mask: " + str(input_mask))

        return self.get_sent()


if __name__ == '__main__':
    inference = BertInference('cluener', BERT_MODEL_PATH)

    to_be_pred = '1945年8月斯坦福大学计算机学院阿克琉斯，想去雅典和爱琴海。中国与欧盟海军爱玩dota'
    model = BertSoftmaxForNer.from_pretrained(KNLP_PATH + '/knlp/model/bert/output_modelbert/checkpoint-448')
    model.to('cpu')
    result = inference.predict(to_be_pred, model)
    print(result)
    print(inference.get_entity())
