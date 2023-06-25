# Importing Necessary modules
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.sentiment_pipeline import SentimentPipeline
import uvicorn

app = FastAPI()


# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'TOXIC SENTIMENT ANALYSIS!'}


class RequestBody(BaseModel):
    text: str


def load_sentiment_pipeline():
    sent_pipeline = SentimentPipeline.build()
    return sent_pipeline


@app.post('/predict')
def predict(data: RequestBody):
    input_text = data.text
    pipeline = load_sentiment_pipeline()
    dp = pipeline.analyze(input_text)
    result = dp.sentiment
    return {"Sentiment": result}

