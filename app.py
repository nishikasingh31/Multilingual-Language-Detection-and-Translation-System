import streamlit as st
from transformers import pipeline
import pandas as pd
import joblib
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and vectorizer
model = joblib.load("model/language_classifier_model.pkl")
vectorizer = joblib.load("model/language_vectorizer.pkl")

# Title
st.title("Multilingual Language Detection and Translation System")

# Input text
text = st.text_area("Enter text here:", height=150)

# Choose functionality
option = st.selectbox(
    "Choose an action:",
    ["Detect Language", "Translate Text", "Visualize Word Cloud"]
)

# Process user input
if st.button("Run"):
    if option == "Detect Language":
        # Preprocess the input text using the loaded vectorizer
        reshaped_input = vectorizer.transform([text])  # Transform the text into numerical data
        lang_detect = model.predict(reshaped_input)[0]  # Predict language using the model
        st.write(f"Detected Language: {lang_detect}")

    elif option == "Translate Text":
        # Example: Translation logic using a pre-trained transformer model
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
        translation = translator(text, max_length=400)
        st.write(f"Translation: {translation[0]['translation_text']}")

    elif option == "Visualize Word Cloud":
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        # Generate a word cloud
        wordcloud = WordCloud(width=800, height=400).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
