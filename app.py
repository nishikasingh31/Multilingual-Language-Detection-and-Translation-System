import streamlit as st
import langid
import joblib
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from transformers import MarianMTModel, MarianTokenizer

# Load the pre-trained model and vectorizer
model = joblib.load("model/language_classifier_model.pkl")
vectorizer = joblib.load("model/language_vectorizer.pkl")


# Cache the translation model/tokenizer so they load only once per session,
# not on every button click / script rerun.
@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-mul-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, translation_model


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
            # Translation using MarianMT directly (no pipeline(), so no
            # dependency on transformers' task registry / processor auto-detection)
            if text.strip():
                tokenizer, translation_model = load_translation_model()
                inputs = tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                )
                outputs = translation_model.generate(**inputs, max_length=400)
                translated_text = tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                st.write(f"Translation: {translated_text}")
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
