import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(file_path):
    # Load Sentiment140 Dataset
    sentiment140 = pd.read_csv(file_path, encoding='latin-1', header=None, names=['polarity', 'id', 'date', 'query', 'user', 'text'])
    
    # Map polarity: 0 (negative), 2 (neutral), 4 (positive)
    sentiment140['label'] = sentiment140['polarity'].apply(lambda x: 2 if x == 2 else (1 if x == 4 else 0))
    
    # Use only text and label columns
    df_sentiment140 = sentiment140[['text', 'label']]
    
    # Vectorize text data
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(df_sentiment140['text'])
    train_labels = df_sentiment140['label']
    
    return train_vectors, train_labels, vectorizer
