import streamlit as st
import joblib, os
from train_model import train_and_save_model  # Import function

MODEL_FILE = "language_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# Train if model files are missing
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    st.warning("⚠️ Model files missing. Training new model...")
    train_and_save_model()

# Load model
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

st.title("🌍 Language Detection App")
text = st.text_area("Enter text:")
if st.button("Detect"):
    if text.strip():
        X_vec = vectorizer.transform([text])
        pred = model.predict(X_vec)[0]
        st.success(f"Predicted Language: **{pred}**")
