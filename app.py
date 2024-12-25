from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
lemmatizer = WordNetLemmatizer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

def preprocess_reviews(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(all_stopwords)]
    return ' '.join(review)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    input_text = request.form.get('textInput')
    processed_text = preprocess_reviews(input_text)
    text_vector = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(text_vector)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    emoji = '😊' if sentiment == 'Positive' else '😞'
    return render_template('index.html', sentiment=sentiment, emoji=emoji, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
      