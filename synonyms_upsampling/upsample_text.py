import re
import json
from typing import Optional, List
import csv

input_file_txt = '/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_sample_toxic.txt'
input_json = '/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/syn.json'


def gen_text(text: str) -> List[str | None]:
    text_new = []
    gen_text = []
    # hashtags = re.findall(r'\b#\w+', text)
    mapper_texts = []
    hashtags = re.findall(r'#\w+', text)
    l = len(hashtags)
    with open(input_json, "r") as f_json:
        mapper = json.load(f_json)
        for hashtag in hashtags:
            key = hashtag.replace("#", "")
            mapper_texts.append(mapper[key])
        for i in range(l):
            if i == 1:
                return text_new
            for t in mapper_texts[i]:
                gen_text.append(text.replace(hashtags[i], t))
            if l == 1:
                return gen_text
            for gen_t in gen_text:
                for j in mapper_texts[l-1-i]:
                    text_new.append(gen_t.replace(hashtags[l-1-i], j))

    f_json.close()


output_file = "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/texts_gen_toxic.csv"
fields = ['free_text', 'label_id']
text_gens = []
with open(input_file_txt, 'r') as f:
    texts = f.readlines()
    for text in texts:
        try:
            texts_new = set(gen_text(text))
            for t in texts_new:
                text_gens.append([t.replace("\n", "").replace("\r", "").strip('"'), 1])
        except Exception as e:
            print(text)

with open(output_file, "w+") as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(text_gens)



f.close()
