import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AdamW

from knlp.seq_labeling.bert.models.bert_for_ner import BertQueryNER
from knlp.seq_labeling.NER.Inference.Inference import NERInference
from knlp.common.constant import KNLP_PATH
from knlp.utils.metrics.functional.mrc_ner_evaluate import flat_ner_decode
from knlp.utils.mrc_utils import InputExample, InputFeatures
from knlp.utils.get_entity import get_entities
from knlp.utils.tokenization import BasicTokenizer


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default=KNLP_PATH + "/knlp/model/bert/Chinese_wwm", type=str, )
    parser.add_argument("--data_dir", default=KNLP_PATH + "/knlp/data/mrc", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--predict_batch_size", default=3, type=int)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--entity_threshold", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--saved_model", type=str,
                        default=KNLP_PATH + "/knlp/model/bert/mrc_ner/bert_finetune_model_v5.bin")
    parser.add_argument("--weight_start", type=float, default=1.0)
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument("--weight_span", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--entity_sign", type=str, default="flat")
    return parser


class MRCNER_Inference(NERInference):
    def __init__(self, mrc_data_path=None, tokenizer_vocab=None, data_sign=None, log=False):
        super().__init__()
        self.config = get_argparse().parse_args()
        self.config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_dir = mrc_data_path
        self.sign = data_sign
        self.result = []
        self.vocab = KNLP_PATH + '/knlp/model/bert/Chinese_wwm/vocab.txt' if not tokenizer_vocab else tokenizer_vocab
        if log:
            self.detailed_log()

    def get_examples(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        with open(os.path.join(self.data_dir, f'{self.sign.lower()}.json'), 'r') as fp:
            query = json.loads(fp.read())['default']
        examples = []
        for text in texts:
            for k, q in query.items():
                examples.append(
                    InputExample(
                        query_item=q,
                        context_item=text,
                        ner_cate=k
                    )
                )
        return examples

    def predict_convert_examples_to_features(self, examples, tokenizer, label_lst, max_seq_length, pad_sign=True):
        label_map = {tmp: idx for idx, tmp in enumerate(label_lst)}
        features = []
        total = len(examples)
        contents = []
        for (example_idx, example) in enumerate(examples):

            basicTokenizer = BasicTokenizer(vocab_file=self.vocab,
                                            do_lower_case=True)

            whitespace_doc = basicTokenizer.tokenize(example.context_item)
            query_tokens = tokenizer.tokenize(example.query_item)

            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
            all_doc_tokens = []

            for token_item in whitespace_doc:
                tmp_subword_lst = tokenizer.tokenize(token_item)
                all_doc_tokens.extend(tmp_subword_lst)

            if len(all_doc_tokens) >= max_tokens_for_doc:
                all_doc_tokens = all_doc_tokens[: max_tokens_for_doc]

            contents.append(['[CLS]'] + query_tokens + ['[SEP]'] + all_doc_tokens + ['[SEP]'])
            input_tokens = []
            segment_ids = []
            input_mask = []

            input_tokens.append("[CLS]")
            segment_ids.append(0)
            for query_item in query_tokens:
                input_tokens.append(query_item)
                segment_ids.append(0)
            input_tokens.append("[SEP]")
            segment_ids.append(0)
            input_mask.append(1)
            input_tokens.extend(all_doc_tokens)
            segment_ids.extend([1] * len(all_doc_tokens))
            input_tokens.append("[SEP]")
            segment_ids.append(1)
            input_mask = [1] * len(input_tokens)

            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

            if len(input_ids) < max_seq_length and pad_sign:
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
            if self.log:
                if example_idx < 1:
                    print("example.context_item:", example.context_item)
                    print("tokens:", input_tokens)
                    print("input_ids:", input_ids)
                    print("input_mask:", input_mask)
                    print("segment_ids:", segment_ids)
                    print("input_ids:", input_ids)
                    print("ner_cate:", example.ner_cate)
            features.append(
                InputFeatures(
                    tokens=input_tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    ner_cate=label_map[example.ner_cate]
                ))

        return features, contents

    def predict_loader(self, texts, tokenizer):
        with open(os.path.join(self.data_dir, f'{self.sign.lower()}.json'), 'r') as fp:
            label_list = json.loads(fp.read())['labels']

        label_list = label_list + ['O']
        examples = self.get_examples(texts)
        features, contents = self.predict_convert_examples_to_features(
            examples, tokenizer, label_list, self.config.max_seq_length
        )
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        ner_cate = torch.tensor([f.ner_cate for f in features], dtype=torch.long)
        dataset = TensorDataset(input_ids, input_mask, segment_ids, ner_cate)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=self.config.predict_batch_size, num_workers=2)
        return dataloader, label_list, contents

    def load_model(self):
        model = BertQueryNER(self.config)
        checkpoint = torch.load(os.path.join(self.config.output_dir, self.config.saved_model),
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.to(self.config.device)
        return model

    def predict(self, model, predict_dataloader, label_list, tokenizer, contents):
        device = self.config.device
        model.eval()

        self.entity_set.clear()
        self.tag_list.clear()

        start_pred_lst = []
        end_pred_lst = []

        mask_lst = []

        eval_steps = 0
        ner_cate_lst = []

        for input_ids, input_mask, segment_ids, ner_cate in predict_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                start_logits, end_logits = model(input_ids, segment_ids, input_mask)

            start_label = start_logits.detach().cpu().numpy().tolist()
            end_label = end_logits.detach().cpu().numpy().tolist()

            input_mask = input_mask.to("cpu").detach().numpy().tolist()

            ner_cate_lst += ner_cate.numpy().tolist()
            mask_lst += input_mask
            eval_steps += 1

            start_pred_lst += start_label
            end_pred_lst += end_label

        span_pred_lst = [[[1] * len(start_pred_lst[0])] * len(start_pred_lst[0])] * len(start_pred_lst)

        if self.config.entity_sign == "flat":
            pred_bmes_label_lst = flat_ner_decode(start_pred_lst,
                                                  end_pred_lst,
                                                  span_pred_lst,
                                                  ner_cate_lst,
                                                  label_list,
                                                  threshold=self.config.entity_threshold,
                                                  dims=2)
            flag = 0
            for content, pre in zip(contents, pred_bmes_label_lst):
                text = content[content.index('[SEP]') + 1:-1]
                chunks = get_entities(pre, content)
                if flag % 10 == 0:
                    print("".join(text))
                if chunks:
                    self.result.append(chunks)
                    for item in chunks:
                        self.entity_set.add((''.join(item[0]), item[2]))
                flag += 1

    def run(self, words):
        model = self.load_model()
        tokenizer = BertTokenizer.from_pretrained(self.config.bert_model)
        dataloader, label_list, contents = self.predict_loader(words, tokenizer)
        self.predict(model, dataloader, label_list, tokenizer, contents)

        init_tag = ['O' for _ in range(len(words))]
        res = self.get_chunks()
        for contain in res:
            for piece in contain:
                union = ''.join(piece[0])
                label = piece[2]
                begin = words.find(union)
                end = begin + len(union) - 1
                init_tag[begin] = 'B' + '-' + label
                middle_tags = [('I' + '-' + label) for _ in range(end - begin)]
                init_tag[begin + 1:end + 1] = middle_tags

        self.tag_list = init_tag
        print(init_tag)
        print(self.get_entity())

    def get_chunks(self):
        return self.result


if __name__ == '__main__':
    for_pred = "普林斯顿大学和清华大学的阿格琉斯，根据《国际法》，对于土耳其政府的行为表示强烈谴责。爱玩csol"
    test = MRCNER_Inference()
    test.run(for_pred)
