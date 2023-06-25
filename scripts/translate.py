from googletrans import Translator
import googletrans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def translate(text):
    translator = Translator()
    translation = translator.translate(text, src="en", dest="vi")
    return translation.text


df = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/scripts/train.csv")

with open("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/scripts/obsence_text.txt", "w+") as f:
    for index, row in df.iterrows():
        obscene = row.obscene
        if obscene == 1:
            f.writelines(translate(row.comment_text))
f.close()
