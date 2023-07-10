from pipeline.base_obj import PipelineComponent
from pipeline.data_obj.datapoint import DataPoint
import torch
from LSTM.SentimentLSTM import ModelSentimentLSTM
from BERT.phoBERT_sentiment import PhoBERTSentiment, PhoBertBiLSTM, PhoBertCNN


class SentimentAnalysis(PipelineComponent):
    def serve(self, dp: DataPoint):
        input_text = torch.tensor(dp.text_array, dtype=torch.long)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dp.result == 1:
            sent_model = PhoBERTSentiment(num_classes=4)
            state_dict = torch.load("models/sentiment/best_acc_sent_toxic.pth",
                                            map_location=device)
            sent_model.load_state_dict(state_dict)
            output = sent_model(input_text).argmax(dim=1)
            dp.result = output + 1
        return dp


class BiToxicCmtAnalysis(PipelineComponent):
    def serve(selfself, dp: DataPoint):
        input_text = torch.tensor(dp.text_array, dtype=torch.long)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Tắt cảnh báo
        sent_model = PhoBertCNN(2).to(device)
        state_dict = torch.load(
            "models/sentiment/best_cnn_acc_sent_toxic_update_weight_ver2.pth",
            map_location=device)
        sent_model.load_state_dict(state_dict)
        sent_model.to(device)
        output = sent_model(input_text)
        dp.result = output.argmax(dim=1)
        return dp

