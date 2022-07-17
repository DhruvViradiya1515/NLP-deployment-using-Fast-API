# Importing Libraries
import numpy as np
import pandas as pd
# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier as XGB
# text preprocessing modules
from string import punctuation 
# text preprocessing modules(NLTK)
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re #regular expression
# Download dependency
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
):
    nltk.download(dependency)
nltk.download('omw-1.4')
import warnings
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)


# Importing Dataset

data = pd.read_csv("airline_sentiment_analysis.csv")
data = data.drop('Unnamed: 0',axis=1)
stop_words =  stopwords.words('english')



def label_encoding(text):
    if(text == "negative"):
        return 0
    else:
        return 1
data['airline_sentiment'] = data['airline_sentiment'].apply(lambda text: label_encoding(text))

# Class for pre-processing the dataset

class Text_cleaner:
    def __init__(self,data) -> None:
        self.data = data
    def _text_cleaning(self,text, remove_stop_words=True, lemmatize_words=True):
        # Clean the text, with the option to remove stop_words and to lemmatize word
        # Clean the text
        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = re.sub(r"\'s", " ", text)
        text =  re.sub(r'http\S+',' link ', text)
        text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
            
        # Remove punctuation from text
        text = ''.join([c for c in text if c not in punctuation])
        
        # Remove stop words
        if remove_stop_words:
            text = text.split()
            text = [w for w in text if not w in stop_words]
            text = " ".join(text)
        
        # Shorten words to their stems
        if lemmatize_words:
            text = text.split()
            lemmatizer = WordNetLemmatizer() 
            lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
            text = " ".join(lemmatized_words)
        
        # Return a list of words
        return(text)
    
    def text_clean(self):
        X = data["text"].apply(self._text_cleaning)
        y = data["airline_sentiment"].values
        return X,y

# cleaning the dataset

TC = Text_cleaner(data)
X,y = TC.text_clean()
print(X)
# Class for training 

class Sentimental_classifier:
    def __init__(self,vectorizor,base_model):
        self.sentimental_pipeline = Pipeline(steps=[
                                 ('pre_processing',vectorizor),
                                 ('naive_bayes',base_model)
                                 ])
    def fit(self,X,y):
        self.sentimental_pipeline.fit(X,y)

# Fitting the model

bow = TfidfVectorizer(lowercase=False)
base_model = LogisticRegression()

sentiment_classifier = Sentimental_classifier(bow,base_model)

sentiment_classifier = Pipeline(steps=[
                                 ('pre_processing',bow),
                                 ('naive_bayes',base_model)
                                 ])
sentiment_classifier.fit(X,y)
# print(sentiment_classifier.predict([a1]))

# Saving the model

import joblib
joblib.dump(sentiment_classifier,"sent_model.pkl")