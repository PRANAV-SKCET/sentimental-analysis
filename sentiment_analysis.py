import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

# Load Sentiment140 Dataset
file_path = './sentiment140.csv'  # Replace with your file path
sentiment140 = pd.read_csv(file_path, encoding='latin-1', header=None, names=['polarity', 'id', 'date', 'query', 'user', 'text'])

# Map polarity: 0 (negative), 2 (neutral), 4 (positive)
sentiment140['label'] = sentiment140['polarity'].apply(lambda x: 2 if x == 2 else (1 if x == 4 else 0))

# Use only text and label columns
df_sentiment140 = sentiment140[['text', 'label']]

# Preprocess and Vectorize
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(df_sentiment140['text'])
train_labels = df_sentiment140['label']

# Train Naive Bayes Model
nb_classifier = MultinomialNB()
nb_classifier.fit(train_vectors, train_labels)

# Load BERT for sarcasm detection
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sarcasm_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
sarcasm_pipeline = pipeline("text-classification", model=sarcasm_model, tokenizer=tokenizer)

# Define prediction functions
def get_sentiment_nb(text, threshold=0.7):
    text_vector = vectorizer.transform([text])
    prediction_proba = nb_classifier.predict_proba(text_vector)
    max_proba = max(prediction_proba[0])
    if max_proba >= threshold:
        return nb_classifier.predict(text_vector)[0]  # 0 (negative), 1 (positive), 2 (neutral)
    else:
        return -1  # Indicate low confidence for neutral

def get_sarcasm_bert(text):
    prediction = sarcasm_pipeline(text)[0]
    return 1 if prediction['label'] == 'LABEL_1' else 0

def combine_predictions(text):
    nb_prediction = get_sentiment_nb(text)
    if len(text.split()) <= 3:
        return nb_prediction if nb_prediction != -1 else 2  # Assume neutral if uncertain

    sarcasm_prediction = get_sarcasm_bert(text)
    if nb_prediction == -1:
        return sarcasm_prediction
    return nb_prediction if nb_prediction == sarcasm_prediction else sarcasm_prediction

# User Interface
while True:
    print("\nEnter a sentence for analysis (type 'exit' to quit):")
    user_input = input()
    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    combined_result = combine_predictions(user_input)
    sentiment_label = "Positive" if combined_result == 1 else "Negative" if combined_result == 0 else "Neutral"
    print(f"Combined Sentiment Prediction: {sentiment_label}")
