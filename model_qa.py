#https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
#https://www.youtube.com/watch?v=ovD_87gHZO4&t=811s
from transformers import AutoModel, modeling_outputs, BertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F


class CNNLayer(nn.Module):
    def __init__(self, seq_len, embed_size):
        super().__init__()
        #TODO1:현재 발표에 나와있는대로 seq방향이 아닌 히든 임베딩 방향으로 콘볼루션하고 있는데, seq방향으로 테스트 해봐야함.
        self.conv1d_k3 = nn.Conv1d(in_channels = seq_len, out_channels = seq_len * 2, kernel_size = 3, padding = 1)
        self.conv1d_k1 = nn.Conv1d(in_channels = seq_len * 2, out_channels = seq_len, kernel_size = 1)
        #TODO2:[seq_len, 임베딩 사이즈] -> 임베딩 사이즈로 실험
        self.layer_norm = nn.LayerNorm([seq_len, embed_size])
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1d_k3(x)
        out = self.conv1d_k1(out)
        out = F.relu(out)
        out = out + x
        out = self.layer_norm(out)
        out = self.drop_out(out)

        return out
        


class BertForQuestionAnsweringwithCNN(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]


    def __init__(self, model_name, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = AutoModel.from_pretrained(model_name, config = config, add_pooling_layer=False)
        self.cnnLayers = nn.ModuleList([CNNLayer(config.max_position_embeddings, config.hidden_size) for _ in range(5)])
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        start_positions = None,
        end_positions = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        for layer in self.cnnLayers:
            sequence_output = layer(sequence_output)
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return modeling_outputs.QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )