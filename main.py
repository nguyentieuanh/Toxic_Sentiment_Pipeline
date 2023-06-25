from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from dataloader.textloader import from_text2array
from common_utils.text_utils import preprocessing
from pipeline.sentiment_pipeline import SentimentPipeline
import torch


#
# class BPE:
#     bpe_codes = 'PhoBERT_base_transformers/bpe.codes'
#
#
# args = BPE()
# bpe = fastBPE(args)
#
# # # Load the dictionary
# vocab = Dictionary()
# vocab.add_from_file("PhoBERT_base_transformers/dict.txt")

def load_sentiment_pipeline():
    sent_pipeline = SentimentPipeline.build()
    return sent_pipeline


input_text = "bọn m óc chó à"

pipeline = load_sentiment_pipeline()
dp = pipeline.analyze(input_text)
print(dp.sentiment)
# text = from_text2array(preprocessing(input_text), vocab, bpe, 50)
# print(text.shape)
