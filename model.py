import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
from transformers import PreTrainedModel, RobertaModel, RobertaConfig


class BertABSATagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        bert_config: configuration for bert model
        """
        super(BertABSATagger, self).__init__(bert_config)
        self.num_labels = bert_config.num_labels

        # initialized with pre-trained BERT and perform finetuning
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        # hidden size at the penultimate layer
        penultimate_hidden_size = bert_config.hidden_size            
        self.classifier = nn.Linear(penultimate_hidden_size, bert_config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        tagger_input = outputs[0]
        tagger_input = self.bert_dropout(tagger_input)
        # print("tagger_input.shape:", tagger_input.shape)

        logits = self.classifier(tagger_input)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            # print("We are using true labels!")
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # use soft labels to train the model
        if teacher_probs is not None:
            # print("We are using soft labels!")
            loss_kd_func = MSELoss(reduction='none')
            active_loss = attention_mask.view(-1) == 1

            pred_probs = torch.nn.functional.softmax(logits, dim=-1)
            loss_kd = loss_kd_func(pred_probs, teacher_probs)  # batch_size x max_seq_len x num_labels

            loss_kd = torch.mean(loss_kd.view(-1, self.num_labels)[active_loss.view(-1)])

            outputs = (loss_kd,) + outputs

        return outputs


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple 
    interface for downloading and loading pretrained models.
    """
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class XLMRABSATagger(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None):

        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        tagger_input = outputs[0]
        tagger_input = self.dropout(tagger_input)
        # print("tagger_input.shape:", tagger_input.shape)

        logits = self.classifier(tagger_input)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            # print("We are using true labels!")
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # use soft labels to train the model
        if teacher_probs is not None:
            # print("We are using soft labels!")
            loss_kd_func = MSELoss(reduction='none')
            active_loss = attention_mask.view(-1) == 1

            pred_probs = torch.nn.functional.softmax(logits, dim=-1)
            loss_kd = loss_kd_func(pred_probs, teacher_probs)  # batch_size x max_seq_len x num_labels

            loss_kd = torch.mean(loss_kd.view(-1, self.num_labels)[active_loss.view(-1)])

            outputs = (loss_kd,) + outputs

        return outputs
