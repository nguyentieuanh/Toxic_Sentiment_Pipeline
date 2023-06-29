from pipeline.base_obj import PipelineComponent
from pipeline.data_obj.datapoint import DataPoint
import torch
from LSTM.SentimentLSTM import ModelSentimentLSTM
from BERT.phoBERT_sentiment import PhoBERTSentiment, PhoBertBiToxicSentiment


class SentimentAnalysis(PipelineComponent):
    def serve(self, dp: DataPoint):
        input_text = torch.tensor(dp.text_array, dtype=torch.long)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dp.result == 1:
            sent_model = PhoBERTSentiment(num_classes=4)
            state_dict = torch.load("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/models/sentiment/best_acc_sent_toxic.pth",
                                            map_location=device)
            sent_model.load_state_dict(state_dict)
            output = sent_model(input_text)
            dp.result = output + 1
        return dp


class BiToxicCmtAnalysis(PipelineComponent):
    def serve(selfself, dp: DataPoint):
        input_text = torch.tensor(dp.text_array, dtype=torch.long)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sent_model = PhoBertBiToxicSentiment()
        output = sent_model(input_text)
        dp.result = output
        return dp

