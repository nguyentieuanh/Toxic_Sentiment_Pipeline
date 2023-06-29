import streamlit as st
from pipeline.sentiment_pipeline import SentimentPipeline


# Loading saved model
def load_sentiment_pipeline():
    sent_pipeline = SentimentPipeline.build()
    return sent_pipeline


# Creating a function for Prediction
def toxic_prediction(input_text):
    pipeline = load_sentiment_pipeline()
    dp = pipeline.analyze(input_text)
    return f'Sắc thái của câu: {dp.sentiment}'


def main():
    # Giving a title
    st.title("Toxic Comment Sentiment Analysis Wep App")

    # getting the input data from the user
    text = st.text_input("Input Text")

    # code for the prediction
    result = ""
    if st.button("Sentiment Result"):
        result = toxic_prediction(text)
    st.success(result)


if __name__ == '__main__':
    main()
