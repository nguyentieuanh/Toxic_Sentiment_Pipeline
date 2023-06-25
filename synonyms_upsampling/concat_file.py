import pandas as pd

train_data = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/train.csv")

gen_text = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/texts_gen_toxic.csv")

train_new_data = pd.concat([train_data, gen_text])
train_new_data.to_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/train_upsampling.csv")
