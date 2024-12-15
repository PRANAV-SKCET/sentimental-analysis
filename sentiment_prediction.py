from transformers import pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def get_sarcasm_pipeline():
    sarcasm_pipeline = pipeline("text-classification", model="bert-base-uncased", tokenizer="bert-base-uncased")
    return sarcasm_pipeline

def get_sentiment_nb(text, vectorizer, nb_classifier, threshold=0.7):
    # Vectorize text for Naive Bayes
    text_vector = vectorizer.transform([text])
    prediction_proba = nb_classifier.predict_proba(text_vector)
    max_proba = max(prediction_proba[0])
    if max_proba >= threshold:
        return nb_classifier.predict(text_vector)[0], max_proba
    return -1, max_proba  # Indicate low confidence for neutral

def get_sarcasm_bert(text, sarcasm_pipeline, sarcasm_threshold=0.9):
    # Run sarcasm detection pipeline
    prediction = sarcasm_pipeline(text)[0]
    confidence = prediction['score']
    is_sarcastic = 1 if prediction['label'] == 'LABEL_1' else 0
    return is_sarcastic, confidence

def combine_predictions(text, vectorizer, nb_classifier, sarcasm_pipeline):
    # Get predictions from both models
    nb_prediction, nb_confidence = get_sentiment_nb(text, vectorizer, nb_classifier)
    sarcasm_prediction, sarcasm_confidence = get_sarcasm_bert(text, sarcasm_pipeline)

    # Logic to prioritize sarcasm if detected with high confidence
    if sarcasm_prediction == 1 and sarcasm_confidence >= 0.8:
        # Sarcasm is prioritized if detected with high confidence
        return 0, sarcasm_confidence, True  # Return negative if sarcastic
    
    # Otherwise, rely on Naive Bayes prediction with a fallback for low confidence
    final_prediction = nb_prediction if nb_prediction != -1 else 2
    return final_prediction, nb_confidence, False
