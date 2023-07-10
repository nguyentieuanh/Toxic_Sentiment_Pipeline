import numpy as np
from tqdm import tqdm
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary


class BPE:
    bpe_codes = 'PhoBERT_base_transformers/bpe.codes'


# Convert df to [N_samples, Max_sequence_length]
def from_csv2array(df, vocab, bpe, max_sequence_length):
    outputs = np.zeros((len(df), max_sequence_length))

    cls_id = 0  # id of the beginning character
    eos_id = 2  # id for the ending character
    pad_id = 1  # id for padding character
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # print(len(df))
        subwords = bpe.encode("<s>" + row.free_text + "</s>")
        input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        if len(input_ids) > max_sequence_length:
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ] * (max_sequence_length - len(input_ids))
        outputs[idx - 1, :] = np.array(input_ids)
        if int(idx) == int(len(df)):
            return outputs
    return outputs


def from_text2array(text, vocab, bpe, max_sequence_length):
    output = np.zeros((1, max_sequence_length))
    cls_id = 0  # id of the beginning character
    eos_id = 2  # id for the ending character
    pad_id = 1  # id for padding character
    subwords = bpe.encode("<s>" + text + "</s>")
    input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
    if len(input_ids) > max_sequence_length:
        input_ids = input_ids[:max_sequence_length]
        input_ids[-1] = eos_id
    else:
        input_ids = input_ids + [pad_id, ] * (max_sequence_length - len(input_ids))
    output[0, :] = np.array(input_ids)
    return output


def load_preprocessing_data(df):
    args = BPE()
    bpe = fastBPE(args)

    # # Load the dictionary
    vocab = Dictionary()
    vocab.add_from_file("PhoBERT_base_transformers/dict.txt")
    x = from_csv2array(df, vocab, bpe, 50)
    y = df.label_id.values.astype("int")
    return x, y
