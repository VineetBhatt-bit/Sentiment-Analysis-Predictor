import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Dataset
df = pd.read_csv("Reviews_cleaned.csv")

print(df.head())

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
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

# Apply preprocessing
df['cleaned_review'] = df['Sentiment'].apply(preprocess)

print(df[['Sentiment', 'cleaned_review']].head())

# Features and Labels
X = df['cleaned_review']
y = df['Sentiment']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)

X_vectorized = tfidf.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save TF-IDF Vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save Model
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel and Vectorizer Saved Successfully!")
