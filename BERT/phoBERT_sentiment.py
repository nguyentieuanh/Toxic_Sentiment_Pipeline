from pipeline.component.pre import BPE
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.models import roberta
import torch.nn as nn
from transformers import RobertaConfig, BertPreTrainedModel, AutoModel, RobertaModel
import torch
from fairseq.models.fairseq_encoder import FairseqEncoder


class PhoBERTSentiment(nn.Module):
    # Load the model in fairseq
    def __init__(self, num_classes):
        super().__init__()
        # pho_bert = AutoModel.from_pretrained("vinai/phobert-base")
        pho_bert = roberta.RobertaModel.from_pretrained("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/PhoBERT_base_fairseq",
                                                checkpoint_file="model.pt")
        args = BPE()
        pho_bert.bpe = fastBPE(args)
        pho_bert.register_classification_head("sentiment_analysis", num_classes=num_classes)
        self.base_model = pho_bert
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = self.base_model.predict("sentiment_analysis", x)
        outputs = self.softmax(outputs).argmax(dim=1)
        return outputs


class RobertaForToxicCmt(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(4 * config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        cls_output = torch.cat((outputs[2][-1][:, 0, ...], outputs[2][-2][:, 0, ...], outputs[2][-3][:, 0, ...],
                                outputs[2][-4][:, 0, ...]), -1)
        logits = self.qa_outputs(cls_output)
        return logits


class PhoBertBiToxicSentiment(nn.Module):
    def __init__(self):
        super().__init__()
        config = RobertaConfig.from_pretrained(
            "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/PhoBERT_base_transformers/config.json",
            output_hidden_states=True,
            num_labels=1
        )

        model_bert = RobertaForToxicCmt.from_pretrained(
            "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/models/bi_cls/model_2.bin",
            config=config)
        self.base_model = model_bert
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = self.base_model(x)
        outputs = int(self.sigmoid(outputs) > 0.5)
        return outputs
