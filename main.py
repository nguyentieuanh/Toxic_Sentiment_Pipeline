from pipeline.sentiment_pipeline import SentimentPipeline
from pipeline.component.layout import SentimentAnalysis
import click


@click.group()
def main():
    pass


def load_sentiment_pipeline():
    sent_pipeline = SentimentPipeline.build()
    return sent_pipeline


def load_model_sentiment_multi_label():
    sent_model = SentimentAnalysis()


@main.command()
@click.option("-in", "--input_text", default="thời tiết đẹp")
def test_one_sentence(input_text):
    pipeline = load_sentiment_pipeline()
    dp = pipeline.analyze(input_text)
    print(dp.sentiment)


@main.command()
@click.option("-in", "--input_folder", default="/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler")
def test_crawler_youtube_sentence(input_folder):
    import os
    pipeline = load_sentiment_pipeline()
    with open(os.path.join(input_folder, "text_crawler.txt"), "r") as f:
        data = f.readlines()
        for text in data:
            text = text.split("\n")[0]
            result = pipeline.analyze(text).sentiment
            print(text, result)


if __name__ == "__main__":
    main()

