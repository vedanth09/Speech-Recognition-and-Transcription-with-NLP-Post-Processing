import os
import requests
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from bertopic import BERTopic
from dotenv import load_dotenv  

# downloading the required nltk resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Loading environment variables from the .env file
load_dotenv()

# Retrieving API_KEY and API_URL from the environment
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

# Function to transcribe audio using AssemblyAI
def transcribe_audio(audio_file_path):
    upload_url = f"{API_URL}/upload"
    headers = {'authorization': API_KEY}
    with open(audio_file_path, 'rb') as audio_file:
        response = requests.post(upload_url, headers=headers, files={'file': audio_file})

    if response.status_code != 200:
        print("Error uploading audio:", response.text)
        return None

    audio_url = response.json()['upload_url']

    # Request transcription
    transcript_url = f"{API_URL}/transcript"
    transcript_request = {'audio_url': audio_url}
    transcript_response = requests.post(transcript_url, headers=headers, json=transcript_request)

    if transcript_response.status_code != 200:
        print("Error requesting transcription:", transcript_response.text)
        return None

    transcript_id = transcript_response.json()['id']
    # Polling the transcription status
    while True:
        check_url = f"{API_URL}/transcript/{transcript_id}"
        status_response = requests.get(check_url, headers=headers)
        status = status_response.json()

        if status['status'] == 'completed':
            return status['text']
        elif status['status'] == 'failed':
            print("Transcription failed.")
            return None
        time.sleep(5)

# Sentiment analysis using Hugging Face
def analyze_sentiment(text):
    try:
        # Loading the sentiment analysis pipeline
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
        # Truncate text to fit the model's input size
        max_length = 512 
        truncated_text = text[:max_length]
        # Analyze sentiment
        sentiment = sentiment_analyzer(truncated_text)
        return sentiment
    except Exception as e:
        print(f"Error in Sentiment Analysis: {e}")
        return None

# Text summarization using Hugging Face
def summarize_text(text, max_length=50, min_length=25):
    try:
        # Loading the summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
        # Summarize the text
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error in Text Summarization: {e}")
        return None


# Preprocessing functions: Tokenization and Lemmatization
def preprocess_text(text):
    try:
        # Tokenize the text into words
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join the lemmatized tokens back into a single string
        preprocessed_text = " ".join(lemmatized_tokens)
        return preprocessed_text
    except Exception as e:
        print(f"Error in Text Preprocessing: {e}")
        return text  # Returns original text in case of error

# Main function to process audio and analyze text
def process_audio(audio_file_path):
    print("Transcribing audio...")
    transcription = transcribe_audio(audio_file_path)

    if transcription:
        print("\n--- Transcription ---")
        print(transcription)

        # Preprocess the transcription before performing NLP tasks
        preprocessed_transcription = preprocess_text(transcription)

        print("\nPerforming sentiment analysis...")
        sentiment = analyze_sentiment(preprocessed_transcription)
        print("Sentiment:", sentiment)

        print("\nGenerating summary...")
        summary = summarize_text(preprocessed_transcription)
        print("Summary:", summary)


if __name__ == "__main__":
    audio_file_path = "/Users/vedanth/Downloads/trump_farewell_address.mp3" 
    process_audio(audio_file_path)
