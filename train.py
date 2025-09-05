import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    data = {
        "text": ["Hello", "Bonjour", "Hola", "नमस्ते", "வணக்கம்"],
        "language": ["English", "French", "Spanish", "Hindi", "Tamil"]
    }
    df = pd.DataFrame(data)

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(df["text"])

    model = RandomForestClassifier()
    model.fit(X_vec, df["language"])

    joblib.dump(model, "language_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("✅ Model trained and saved!")
