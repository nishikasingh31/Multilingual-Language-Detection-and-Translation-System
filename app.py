import streamlit as st
import langid
from transformers import pipeline
import joblib
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

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
    try:
        if option == "Detect Language":
            # Language detection using langid
            if text.strip():
                lang_detect = langid.classify(text)[0]  # Returns (language, confidence)
                st.write(f"Detected Language: {lang_detect}")
            else:
                st.warning("Please enter some text to detect the language.")

        elif option == "Translate Text":
            # Translation using Hugging Face pipeline
            if text.strip():
                translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
                translation = translator(text, max_length=400)
                st.write(f"Translation: {translation[0]['translation_text']}")
            else:
                st.warning("Please enter some text to translate.")

        elif option == "Visualize Word Cloud":
            # Word Cloud visualization
            if text.strip():
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    stopwords=STOPWORDS
                ).generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.warning("Please enter some text to generate a word cloud.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
