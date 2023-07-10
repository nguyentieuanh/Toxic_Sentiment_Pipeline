import pandas as pd
from common_utils.text_utils import preprocessing
from tqdm import tqdm

tqdm.pandas()


def load_file_csv(file_path):
    df = pd.read_csv(file_path)
    df = df[df["free_text"].notna()]
    df = df.replace(2, 1)
    df['free_text'] = df['free_text'].progress_apply(lambda x: preprocessing(x))
    return df


def load_file_csv_bi_toxic(file_path):
    df = pd.read_csv(file_path)
    df = df[df["free_text"].notna()]
    df = df.drop_duplicates()
    df = df.replace(2, 1)
    df["free_text"] = df["free_text"].progress_apply(lambda x: preprocessing(x))
    # df.to_csv("dataset/train/train_data_ver4.csv")
    return df
