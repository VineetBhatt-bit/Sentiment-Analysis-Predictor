import pickle

# Load selector
with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

# Load model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

print("===== Project Information =====")
print("Feature Extraction:", selector["features"])
print("Max Features:", selector["max_features"])

print("\nModel Loaded Successfully!")
print("Vectorizer Loaded Successfully!")
