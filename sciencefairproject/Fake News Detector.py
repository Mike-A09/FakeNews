import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

nltk.download('stopwords')

def load_data():
    df = pd.read_csv('fake_or_real_news.csv')
    return df


def preprocess_data(df):
    df = df[['text', 'label']].dropna()
    df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})
    return df


def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    model = make_pipeline(TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english')), MultinomialNB())

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'fake_news_model.pkl')
    print("Model saved as fake_news_model.pkl")

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    train_model(df)
