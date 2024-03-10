from transformers import AutoModel
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder
from transformers.modeling_outputs import TokenClassifierOutput
from torch import nn


class NERLLM(nn.Module):

    def __init__(self, llm_name, encoder_config, num_labels):

        super().__init__()

        self.llm = AutoModel.from_pretrained(llm_name)


        self.encoder = DebertaV2Encoder(encoder_config)

        self.dropout = nn.Dropout(encoder_config.hidden_dropout_prob)
        self.linear = nn.Linear(encoder_config.hidden_size, num_labels)
        self.num_labels = num_labels

        self._init(self.linear, encoder_config.initializer_range)

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

    def forward(self, input_ids=None, attention_mask=None, labels=None):

        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        sequence_output = self.encoder(sequence_output, attention_mask)

        logits = self.linear(self.dropout(sequence_output))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=None, attentions=None
        )