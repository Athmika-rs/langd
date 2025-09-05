# app.py
import streamlit as st
import joblib
import os
import numpy as np

# ✅ Load model and vectorizer safely
MODEL_FILE = "language_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    st.error("❌ Model files not found! Please run train.py first to generate .pkl files.")
else:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    # ✅ Streamlit Page Config with Theme
    st.set_page_config(
        page_title="🌍 Language Detection App",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Sidebar for theme toggle
    st.sidebar.title("🎨 Theme Settings")
    theme_choice = st.sidebar.radio("Choose Theme:", ["System Default", "Light Mode 🌞", "Dark Mode 🌙"])

    if theme_choice == "Light Mode 🌞":
        st.markdown(
            """
            <style>
            body { background-color: #ffffff; color: #000000; }
            .stTextArea textarea { background-color: #f0f0f0; color: #000000; }
            </style>
            """,
            unsafe_allow_html=True
        )
    elif theme_choice == "Dark Mode 🌙":
        st.markdown(
            """
            <style>
            body { background-color: #0e1117; color: #fafafa; }
            .stTextArea textarea { background-color: #262730; color: #ffffff; }
            </style>
            """,
            unsafe_allow_html=True
        )

    # ✅ App UI
    st.title("🌍 Language Detection Site")
    st.write("This app detects the **language** of a given sentence using **TF-IDF + Random Forest** trained on a multilingual dataset.")

    # ✅ User Input
    user_input = st.text_area("✍️ Enter a sentence:", "")

    if st.button("Detect Language"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a sentence to detect.")
        else:
            # Transform input
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
            proba = model.predict_proba(input_vec)[0]

            # Get top 3 probable languages
            top3_idx = np.argsort(proba)[-3:][::-1]
            st.success(f"✅ Detected Language: **{prediction}**")

            st.subheader("🔎 Prediction Confidence:")
            for idx in top3_idx:
                st.write(f"**{model.classes_[idx]}** → {proba[idx]*100:.2f}%")
