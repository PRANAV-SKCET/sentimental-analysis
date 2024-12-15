from flask import Flask, render_template, request
from data_processing import load_and_preprocess_data
from model_training import train_naive_bayes_model
from sentiment_prediction import combine_predictions, get_sarcasm_pipeline

# Initialize Flask app
app = Flask(__name__)

# Load data and models
file_path = './sentiment140.csv'
train_vectors, train_labels, vectorizer = load_and_preprocess_data(file_path)
nb_classifier = train_naive_bayes_model(train_vectors, train_labels)
sarcasm_pipeline = get_sarcasm_pipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    text = request.form['text']
    combined_result, confidence, sarcasm_flag = combine_predictions(
        text, vectorizer, nb_classifier, sarcasm_pipeline
    )
    
    # Determine emoji and display label based on the result
    if combined_result == 1:
        emoji = '😊'  # Positive
        sentiment_label = "Positive"
    elif combined_result == 0:
        emoji = '😞'  # Negative
        sentiment_label = "Negative"
    else:
        emoji = '😐'  # Neutral
        sentiment_label = "Neutral"
    
    # Render the result in the web page
    return render_template(
        'index.html',
        sentiment_label=sentiment_label,
        emoji=emoji,
        user_text=text,
        sarcasm_flag=" (Sarcasm Detected)" if sarcasm_flag else ""
    )

if __name__ == '__main__':
    app.run(debug=True)
