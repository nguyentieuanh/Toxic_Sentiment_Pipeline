import pandas as pd

train_data = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/train_toxic_multilabel.csv")

gen_text_label0 = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/valid/valid_gen_toxic_label0.csv")
gen_text_label1 = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/valid/valid_gen_suicide_label1.csv")
gen_text_label2 = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/valid/valid_gen_harrasment_label2.csv")
gen_text_label3 = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/valid/valid_gen_gender_label3.csv")


train_new_data = pd.concat([train_data, gen_text_label0, gen_text_label1, gen_text_label2, gen_text_label3], ignore_index=True)
train_new_data.to_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/valid_multilabel_toxic.csv")
