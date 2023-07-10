from pipeline.sentiment_pipeline import SentimentPipeline
from pipeline.component.layout import SentimentAnalysis
import click


@click.group()
def main():
    pass


def load_sentiment_pipeline():
    sent_pipeline = SentimentPipeline.build()
    return sent_pipeline


def load_sentiment_bi_class():
    sent_model = SentimentPipeline.build()
    return sent_model


def load_model_sentiment_multi_label():
    sent_model = SentimentAnalysis()


@main.command()
@click.option("-in", "--input_text", default="Nhảy thì ng.u Sờ vú đàn ông thì giỏi")
def test_one_sentence(input_text):
    pipeline = load_sentiment_bi_class()
    dp = pipeline.analyze_bi_class(input_text)
    print(dp.sentiment)


@main.command()
@click.option("-in", "--input_folder", default="/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler")
@click.option("-out", "--output_folder", default="/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/test_text_crawler")
def test_crawler_youtube_sentence(input_folder, output_folder):
    import os
    import csv
    comment_texts = []
    pipeline = load_sentiment_bi_class()
    with open(os.path.join(input_folder, "text_crawler.txt"), "r") as f:
        data = f.readlines()
        for text in data:
            text = text.split("\n")[0]
            result = pipeline.analyze_bi_class(text).result.item()
            print(text, result)
            comment_texts.append([text, result])

    with open(os.path.join(output_folder, "test_crawler.csv"), "w+") as csvfile:
        fields = ["free_text", "sentiment"]
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(comment_texts)

@main.command()
@click.option("-in", "--input_file", default="crawler/test_text_crawler/test_new_2.csv")
@click.option("-out", "--output_folder", default="crawler/test_text_crawler")
def test_dev(input_file, output_folder):
    import pandas as pd
    import os
    import csv
    from train import compute_recall, compute_precision, compute_f1, compute_acc_labels
    from tqdm import tqdm

    comment_texts = []
    predicts = []
    pipeline = load_sentiment_bi_class()
    df = pd.read_csv(input_file).replace(2,1)
    df = df[df["label_id"] == 1]
    print(f'Number predicts: {len(df)}')
    texts = df["free_text"].values
    labels = df["label_id"].values
    pbar = tqdm(enumerate(texts), total=len(texts), position=0, leave=True)
    for index, text in pbar:
        text = text.split("\n")[0].strip()
        result = pipeline.analyze_bi_class(text).result.item()
        text_pro = pipeline.analyze_bi_class(text).text
        print(text_pro, result)
        comment_texts.append([text, result])
        predicts.append(result)

    recall = compute_recall(labels, predicts)
    precision = compute_precision(labels, predicts)
    f1 = compute_f1(labels, predicts)
    acc_labels = compute_acc_labels(labels, predicts)
    print(f'Recall {recall}\n'
          f'Precision {precision}\n'
          f'F1-score: {f1}\n'
          f'Acc label 0: {acc_labels[0]}\n'
          f'Acc label 1: {acc_labels[1]}')

    with open(os.path.join(output_folder, "test_result.csv"), "w+") as csvfile:
        fields = ["free_text", "sentiment"]
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(comment_texts)


if __name__ == "__main__":
    main()