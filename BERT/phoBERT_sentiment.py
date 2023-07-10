from pipeline.component.pre import BPE
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.models import roberta
import torch.nn as nn
from transformers import RobertaConfig, BertPreTrainedModel, RobertaModel
import torch
from torch.nn import functional as F
from transformers import logging

logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"


class PhoBERTSentiment(nn.Module):
    # Load the model in fairseq
    def __init__(self, num_classes):
        super().__init__()
        pho_bert = roberta.RobertaModel.from_pretrained("PhoBERT_base_fairseq",
                                                        checkpoint_file="model.pt")
        args = BPE()
        pho_bert.bpe = fastBPE(args)
        pho_bert.register_classification_head("sentiment_analysis", num_classes=num_classes)
        self.base_model = pho_bert
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = self.base_model.predict("sentiment_analysis", x)
        outputs = self.softmax(outputs)
        return outputs


class BERTForEmbeddingCmt(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        embedded = outputs[2][-1]
        return embedded


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


class PhoBertLSTM(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.n_labels = num_labels
        config = RobertaConfig.from_pretrained(
            "PhoBERT_base_transformers/config.json",
            output_hidden_states=True,
            num_labels=self.n_labels
        )

        model_bert = BERTForEmbeddingCmt.from_pretrained(
            "PhoBERT_base_transformers/model.bin",
            config=config)
        self.base_model = model_bert
        # BiLSTM
        self.lstm = nn.LSTM(config.hidden_size, 256, num_layers=2, batch_first=True, dropout=0.2,
                            bidirectional=False)
        self.fc = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.qa_outputs = nn.Linear(128, self.n_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.base_model(x)
        h0 = torch.zeros(1 * 2, embedded.size(0), 256).to(device)
        c0 = torch.zeros(1 * 2, embedded.size(0), 256).to(device)
        out, _ = self.lstm(embedded, (h0, c0))
        # batch_size, seq_length, hidden_size
        out = out[:, -1, :]
        fc = self.fc(out)
        fc_output = self.relu(fc)
        logits = self.qa_outputs(fc_output)
        outputs = self.softmax(logits)
        return outputs


class PhoBertBiLSTM(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.n_labels = num_labels
        config = RobertaConfig.from_pretrained(
            "PhoBERT_base_transformers/config.json",
            output_hidden_states=True,
            num_labels=self.n_labels
        )

        model_bert = BERTForEmbeddingCmt.from_pretrained(
            "PhoBERT_base_transformers/model.bin",
            config=config)
        self.base_model = model_bert
        # BiLSTM
        self.lstm = nn.LSTM(config.hidden_size, 256, num_layers=2, batch_first=True, dropout=0.4,
                            bidirectional=True)
        self.fc = nn.Linear(256 * 2, 128)
        self.relu = nn.ReLU()
        self.qa_outputs = nn.Linear(128, self.n_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.base_model(x)
        h0 = torch.zeros(2 * 2, embedded.size(0), 256).to(device)
        c0 = torch.zeros(2 * 2, embedded.size(0), 256).to(device)
        out, _ = self.lstm(embedded, (h0, c0))
        # batch_size, seq_length, hidden_size
        out = out[:, -1, :]
        fc = self.fc(out)
        fc_output = self.relu(fc)
        logits = self.qa_outputs(fc_output)
        outputs = self.softmax(logits)
        return outputs


class PhoBertCNN(nn.Module):
    def __init__(self, n_labels):
        self.n_labels = n_labels
        super().__init__()
        config = RobertaConfig.from_pretrained("PhoBERT_base_transformers/config.json",
                                               output_hidden_states=True,
                                               num_labels=self.n_labels)
        model_bert = BERTForEmbeddingCmt.from_pretrained(
            "PhoBERT_base_transformers/model.bin",
            config=config)
        self.base_model = model_bert
        # batch_size, seq_length, embedding_dims = config.hidden_size
        # CNN - input = batch_size, embedding_dims, seq_length
        self.conv1d_1 = nn.Conv1d(in_channels=config.hidden_size,
                                  out_channels=512,
                                  kernel_size=7,
                                  device=device)
        self.max_pooling1d_1 = nn.MaxPool1d(kernel_size=2,
                                          stride=2)
        self.conv1d_2 = nn.Conv1d(in_channels=512,
                                  out_channels=256,
                                  kernel_size=5,
                                  device=device)
        self.max_pooling1d_2 = nn.MaxPool1d(kernel_size=2,
                                          stride=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(256*9, 128)
        self.dropout = nn.Dropout(0.4)
        self.linear_2 = nn.Linear(128, self.n_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.base_model(x)
        texts = embedded.permute(0,2,1)
        outputs = self.conv1d_1(texts)
        outputs = self.max_pooling1d_1(outputs)
        outputs = self.conv1d_2(outputs)
        outputs = self.max_pooling1d_2(outputs)
        outputs = self.flatten(outputs)
        outputs = F.relu(self.linear_1(outputs))
        outputs = self.dropout(outputs)
        outputs = self.linear_2(outputs)
        logits = self.softmax(outputs)
        return logits


class PhoBERTMultiToxicCls(nn.Module):
    def __init__(self):
        super().__init__()
        config = RobertaConfig.from_pretrained(
            "PhoBERT_base_transformers/config.json",
            output_hidden_states=True,
            num_labels=4
        )
        model_bert = RobertaForToxicCmt.from_pretrained(
            "PhoBERT_base_transformers/model.bin",
            config=config)
        self.base_model = model_bert
        self.softmax = nn.Softmax()

    def forward(self, x):
        outs = self.base_model(x)
        outputs = self.softmax(outs)
        return outputs
