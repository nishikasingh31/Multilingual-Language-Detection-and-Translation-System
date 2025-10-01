# Multilingual Language Detection & Translation System

https://github.com/user-attachments/assets/0a399755-90c5-4684-a054-22273ef3b809

An interactive **Streamlit web application** that detects the language of input text and provides real-time translation into multiple languages.  
Built with **Python, Streamlit, Hugging Face Transformers, and LangID**, this project demonstrates multilingual NLP capabilities in a simple, user-friendly interface.

## Streamlit Deployment
https://nishikasingh31-multilingual-language-detection-and-translation.streamlit.app/

## 🚀 Features
- 🌍 **Language Detection** — Automatically identifies the input language.  
- 🔄 **Text Translation** — Translates text into multiple supported languages.  
- ⚡ **Real-time Web UI** — Built with Streamlit for quick experimentation.  
- 📊 **Visualization & Insights** — Shows detection results, translation outputs, and optional metrics.  
- 🛠️ **Easy Deployment** — Run locally or host via Streamlit Cloud.  

## 🛠️ Tech Stack
- **Frontend:** Streamlit  
- **Language Detection:** LangID, scikit-learn  
- **Translation Models:** Hugging Face Transformers (e.g., MarianMT, mBART, etc.)  
- **NLP Tools:** NLTK / Tokenizers  
- **Runtime:** Python 3.x  

## ▶️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/nishikasingh31/Multilingual-Language-Detection-and-Translation-System.git
cd Multilingual-Language-Detection-and-Translation-System
```
2. Install dependencies
```bash
Copy code
pip install -r requirements.txt
```
3. Run the Streamlit app
```bash
Copy code
streamlit run app.py
```
4. Open in Browser
```
The app will run locally on:
👉 http://localhost:8501
```
🌍 Usage
-Open the Streamlit interface.
-Enter or paste your text into the input box.
-Click Detect Language → system shows detected language.
-Select a target language from the dropdown.
-Click Translate → see translated output instantly.

📊 Example

| Input Text               | Detected Language | Translated to English |
|--------------------------|-----------------|----------------------|
| "Hola, ¿cómo estás?"     | Spanish         | "Hello, how are you?" |
| "Bonjour tout le monde"  | French          | "Hello everyone"      |
| "मैं ठीक हूँ"             | Hindi           | "I am fine"           |

🚧 Future Enhancements
-Add speech-to-text input for multilingual audio.
-Support more languages via Hugging Face multilingual models.
-Improve translation accuracy with fine-tuned models.
-Deploy as a public web app (Streamlit Cloud / Hugging Face Spaces).
