import pickle
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load trained model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()

    words = word_tokenize(text)

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# User Input
print("===== Sentiment Analysis Predictor =====")

review = input("Enter Review: ")

# Preprocess review
cleaned_review = preprocess(review)

# Convert into TF-IDF vector
review_vector = tfidf.transform([cleaned_review])

# Predict sentiment
prediction = model.predict(review_vector)[0]

# Output Result
print("\nPrediction Result:")
if prediction == "Positive":
    print("Positive Review 😀")

elif prediction == "Neutral":
    print("Neutral Review 😐")

else:
    print("Negative Review 😡")
