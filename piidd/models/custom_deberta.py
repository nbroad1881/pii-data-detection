import random
from functools import partialmethod
from collections.abc import Sequence
from typing import Optional, Tuple, Union

from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.modeling_outputs import BaseModelOutput, TokenClassifierOutput
import transformers
import torch
from torch import nn

def add_layer_drop(layer_drop_prob):

    transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Encoder.forward = partialmethod(forward, layer_drop_prob=layer_drop_prob)


def forward(
    self,
    hidden_states,
    attention_mask,
    output_hidden_states=True,
    output_attentions=False,
    query_states=None,
    relative_pos=None,
    return_dict=True,
    layer_drop_prob=0.0,
):
    if attention_mask.dim() <= 2:
        input_mask = attention_mask
    else:
        input_mask = attention_mask.sum(-2) > 0
    attention_mask = self.get_attention_mask(attention_mask)
    relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    if isinstance(hidden_states, Sequence):
        next_kv = hidden_states[0]
    else:
        next_kv = hidden_states
    rel_embeddings = self.get_rel_embedding()
    output_states = next_kv
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        # Always get layer when not training or the first layer.
        # otherwise, skip the layer with probability layer_drop_prob
        if (not self.training) or (i == 0) or (random.random() > layer_drop_prob):
            if self.gradient_checkpointing and self.training:
                output_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                    output_attentions,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

        if output_attentions:
            output_states, att_m = output_states

        if i == 0 and self.conv is not None:
            output_states = self.conv(hidden_states, output_states, input_mask)

        if query_states is not None:
            query_states = output_states
            if isinstance(hidden_states, Sequence):
                next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
        else:
            next_kv = output_states

        if output_attentions:
            all_attentions = all_attentions + (att_m,)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (output_states,)

    if not return_dict:
        return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
    )



class MultiSampleDropoutClassifier(nn.Module):

    def __init__(self, hidden_size, num_labels):
        super().__init__()

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(hidden_size, num_labels)

        self._init(self.output, 0.02)


    def _init(self, module, std):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    # https://github.com/abhishekkrthakur/long-text-token-classification/blob/8f636ea23b7e1842583581d9cbdbe9f0f54d3191/train.py#L178
    def forward(self, sequence_output, training=False):

        logits1 = self.output(self.dropout1(sequence_output))

        # at inference, no need to average.
        # each logits should be approximately the same
        if training:
            logits2 = self.output(self.dropout2(sequence_output))
            logits3 = self.output(self.dropout3(sequence_output))
            logits4 = self.output(self.dropout4(sequence_output))
            logits5 = self.output(self.dropout5(sequence_output))

            logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        else:
            logits = logits1

        if training:
            return logits1, logits2, logits3, logits4, logits5
        else:
            return logits
        

class MultiSampleDebertaV2ForTokenClassification(DebertaV2PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = MultiSampleDropoutClassifier(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        if labels is not None:
            logits1, logits2, logits3, logits4, logits5 = self.classifier(sequence_output, training=True)

            logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        else:
            logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            
            loss1 = loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            loss3 = loss_fct(logits3.view(-1, self.num_labels), labels.view(-1))
            loss4 = loss_fct(logits4.view(-1, self.num_labels), labels.view(-1))
            loss5 = loss_fct(logits5.view(-1, self.num_labels), labels.view(-1))
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )