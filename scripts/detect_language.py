import pandas as pd
import langid
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    # Load the preprocessed data
    data = pd.read_csv('C:/Users/nishi/Documents/project/Code/data/preprocessed_data.csv')

    # Print available columns for debugging
    print("Available columns:", data.columns)

    # Ensure all values in 'cleaned_text' are strings
    data['cleaned_text'] = data['cleaned_text'].astype(str)

    # Initialize the vectorizer (you can adjust parameters based on your dataset)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    # Vectorize the 'cleaned_text' column (you can use this for model training)
    X = vectorizer.fit_transform(data['cleaned_text'])

    # Classify the language of the cleaned text using langid
    data['detected_language'] = data['cleaned_text'].apply(lambda x: langid.classify(x)[0])

    # Print to check the detected languages
    print("Detected languages:")
    print(data[['cleaned_text', 'detected_language']].head())

    # Save the results to a new CSV file
    data.to_csv('C:/Users/nishi/Documents/project/Code/data/language_detection_data.csv', index=False)

except Exception as e:
    print("An error occurred:", e)


