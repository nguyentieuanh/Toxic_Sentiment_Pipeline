import pandas as pd
import common_utils.text_utils as text_utils


def load_file_txt(file_path):
    key_value_pairs = []
    with open(file_path, '+r') as file:
        texts = file.readlines()
        # print(texts)
        for t in texts:
            # print(t)
            t = t.replace("\n", "")
            try:
                key, value = t.split("\t")
            except Exception as e:
                print(t.split("\t"))
            key_value_pairs.append((key, value))
        # print(key_value_pairs)

    file.close()
    return key_value_pairs


def load_file_csv(file_path):
    df = pd.read_csv(file_path)
    df = df[df["free_text"].notna()]
    df['free_text'] = df['free_text'].apply(lambda x: text_utils.preprocessing(x))
    return df


def load_file_csv_bi_toxic(file_path):
    df = pd.read_csv(file_path)
    df = df[df["free_text"].notna()]
    df = df.drop_duplicates()
    df = df.replace(2,1)
    df["free_text"] = df["free_text"].apply(lambda x: text_utils.preprocessing(x))
    return df
