# Importing Libraries 

from string import punctuation
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  

import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI

## Declaring App using FastAPI and loading the pickle model

app = FastAPI()
with open(
    join(dirname(realpath(__file__)), "sent_model.pkl"), "rb"
) as f:
    model = joblib.load(f)


## Function for cleaning the dataset

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers

    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:

        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # Return a list of words
    return text

## Predict-sentiment API

@app.get("/predict-review")
def predict_sentiment(review: str):
    # clean the review
    cleaned_review = text_cleaning(review)

    # perform prediction
    prediction = model.predict([cleaned_review])
    output = int(prediction[0])

    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}

    # show results
    result = {"prediction": sentiments[output]}

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)