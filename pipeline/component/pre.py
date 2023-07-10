from pipeline.base_obj import PreProcessComponent
from pipeline.data_obj.datapoint import DataPoint
from typing import Optional
from common_utils.text_utils import preprocessing
from dataloader.textloader import from_text2array

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import warnings
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class BPE:
    bpe_codes = 'PhoBERT_base_transformers/bpe.codes'


class LoadDataComponent(PreProcessComponent):
    def serve(self, text: str) -> Optional[DataPoint]:
        text = preprocessing(text)
        args = BPE()
        bpe = fastBPE(args)

        # # Load the dictionary
        vocab = Dictionary()
        vocab.add_from_file("PhoBERT_base_transformers/dict.txt")
        text2array = from_text2array(text, vocab, bpe, 50)
        len_text = len(text.split())
        quality = 0
        if len_text > 0:
            quality = 1
        dp = DataPoint(
            text_array=text2array,
            result=text2array,
            len_text=len_text,
            text=text,
            quality=quality,
            vocab_size=len(vocab)
        )
        return dp
