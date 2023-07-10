import collections
import pandas as pd


def count_most_common_words(sentences, n):
    words = []
    for sentence in sentences:
        try:
            words.extend(sentence.split())
        except Exception as e:
            continue

    counter = collections.Counter(words)
    most_common = counter.most_common(n)
    return most_common


# Ví dụ sử dụng
df_train = pd.read_csv("synonyms_upsampling/vihsd/train.csv")
sentences = df_train["free_text"].values
# sentences = ["This is the first sentence.", "This is the second sentence.", "And this is the third sentence."]
n = 1000  # Số lượng từ xuất hiện nhiều nhất cần đếm

most_common_words = count_most_common_words(sentences, n)
print(most_common_words)
