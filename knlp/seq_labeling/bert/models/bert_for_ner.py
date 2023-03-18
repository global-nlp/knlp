import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss

from knlp.common.constant import KNLP_PATH
from knlp.seq_labeling.bert.losses.focal_loss import FocalLoss
from knlp.seq_labeling.bert.losses.label_smoothing import LabelSmoothingCrossEntropy


class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertQueryNER(nn.Module):
    def __init__(self, config):
        super(BertQueryNER, self).__init__()
        # bert_config = BertConfig.from_pretrained(config.bert_model)
        # self.bert = BertModel(bert_config)

        self.start_outputs = nn.Linear(config.hidden_size, 2)
        self.end_outputs = nn.Linear(config.hidden_size, 2)

        self.hidden_size = config.hidden_size
        self.bert = BertModel.from_pretrained(KNLP_PATH + '/knlp/model/bert/Chinese_wwm')
        self.loss_wb = config.weight_start
        self.loss_we = config.weight_end

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None):
        """
        Args:
            start_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            end_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
        """

        # print(self.bert(input_ids, token_type_ids, attention_mask))
        # encode, sequence_output = self.bert(input_ids, token_type_ids, attention_mask)
        encode = self.bert(input_ids, token_type_ids, attention_mask)['last_hidden_state']
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)['pooler_output']
        # print(encode[-1].shape)
        # sequence_heatmap = encode[-1]  # batch x seq_len x hidden
        # print(encode.shape)
        sequence_heatmap = encode  # batch x seq_len x hidden
        start_logits = self.start_outputs(sequence_heatmap)  # batch x seq_len x 2
        end_logits = self.end_outputs(sequence_heatmap)  # batch x seq_len x 2


        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()
            # print(start_logits.shape)
            # print(start_positions.shape)
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            total_loss = self.loss_wb * start_loss + self.loss_we * end_loss
            return total_loss
        else:
            start_logits = torch.argmax(start_logits, dim=-1)
            end_logits = torch.argmax(end_logits, dim=-1)

            return start_logits, end_logits
