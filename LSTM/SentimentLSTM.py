import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelSentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dims, hidden_size, num_layers):
        super(ModelSentimentLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        # x -> batch_size, seq_len, feature_size (64, 50, 128)
        self.lstm = nn.LSTM(embedding_dims, hidden_size, num_layers, batch_first=True)
        # (64, 50, 64)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 3)

        # x -> (batch_size, seq_len, feature_size)

    def forward(self, x):
        # initial hidden state
        # h0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size).to(device)
        embedded = self.embedding(x)
        h0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size).to(device)
        # print(embedded.shape)
        out, _ = self.lstm(embedded, (h0, c0))
        # #out = batch size, seq_length, hidden_size
        # #out = (100, 50, 128)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        return out
