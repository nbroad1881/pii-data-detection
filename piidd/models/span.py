from typing import Optional, Tuple, Union

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    StableDropout,
    SequenceClassifierOutput,
)
from transformers.models.deberta.modeling_deberta import (
    DebertaPreTrainedModel,
    DebertaModel,
    StableDropout as StableDropoutV1,
)
import torch
from torch import nn


class DebertaV2ForSpanClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, start_token_id, end_token_id):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)

        self.classifier = nn.Linear(config.hidden_size, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]

        # https://chat.openai.com/share/25feb961-1794-46ef-afdd-91fa10080fda
        bs, seq, hidden = encoder_layer.size()

        range_tensor = (
            torch.arange(seq, device=encoder_layer.device).unsqueeze(0).expand(bs, -1)
        )

        start_mask = (input_ids == self.start_token_id).int().argmax(1).unsqueeze(1)
        end_mask = (input_ids == self.end_token_id).int().argmax(1).unsqueeze(1)

        mask = (range_tensor >= start_mask) & (range_tensor <= end_mask)

        mask = mask.unsqueeze(-1).expand(-1, -1, hidden)

        sum_over_span = (encoder_layer * mask).sum(dim=1)
        count_over_span = mask.sum(dim=1)
        average_over_span = sum_over_span / count_over_span.where(
            count_over_span != 0, torch.ones_like(count_over_span)
        )

        logits = self.classifier(self.dropout(average_over_span))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaV1ForSpanClassification(DebertaPreTrainedModel):
    def __init__(self, config, start_token_id, end_token_id):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config)

        self.classifier = nn.Linear(config.hidden_size, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]

        # https://chat.openai.com/share/25feb961-1794-46ef-afdd-91fa10080fda
        bs, seq, hidden = encoder_layer.size()

        range_tensor = (
            torch.arange(seq, device=encoder_layer.device).unsqueeze(0).expand(bs, -1)
        )

        start_mask = (input_ids == self.start_token_id).int().argmax(1).unsqueeze(1)
        end_mask = (input_ids == self.end_token_id).int().argmax(1).unsqueeze(1)

        mask = (range_tensor >= start_mask) & (range_tensor <= end_mask)

        mask = mask.unsqueeze(-1).expand(-1, -1, hidden)

        sum_over_span = (encoder_layer * mask).sum(dim=1)
        count_over_span = mask.sum(dim=1)
        average_over_span = sum_over_span / count_over_span.where(
            count_over_span != 0, torch.ones_like(count_over_span)
        )

        logits = self.classifier(self.dropout(average_over_span))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )