import os 
import sys
from transformers import pipeline 
from textblob import TextBlob
from keybert import KeyBERT
from concurrent.futures import ThreadPoolExecutor
from mic_input import AudioTranscriber


# Loading the model locally 
_emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=True, device=-1)  # Use device=-1 for CPU and 0 for GPU

class EmotionClassifier:
    """
    Parallel processing of:
    1. Emotion detection using HuggingFace Transformers.
    2. Sentiment analysis using TextBlob.
    3. Keyword extraction using KeyBERT.
    """
    @staticmethod
    def get_emotion_classifier(text):
        try:
            emotion_classifier = _emotion_classifier
            return emotion_classifier(text)
        except Exception as e:
            print(f"Error in emotion classifier: {e}")
            return None

    @staticmethod
    def sentiment_analysis(text):
        """
        1. Polarity: Ranges from -1 (negative) to 1 (positive).
        2. Subjectivity: Ranges from 0 (objective) to 1 (subjective)."""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            return {
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return None

    @staticmethod
    def extract_keywords(text):
        try:
            kw_model = KeyBERT()
            return kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
        except Exception as e:
            print(f"Error in keyword extraction: {e}")
            return None
        
    @staticmethod
    def save_to_dictionary(text):
        """
        Save the results of emotion detection, sentiment analysis, and keyword extraction to a dictionary.
        """
        results = {
            "emotions": EmotionClassifier.get_emotion_classifier(text),
            "sentiment": EmotionClassifier.sentiment_analysis(text),
            "keywords": EmotionClassifier.extract_keywords(text)
        }
        return results
    
def run_all_parallel(text):
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            'emotions': executor.submit(EmotionClassifier.get_emotion_classifier, text),
            'sentiment': executor.submit(EmotionClassifier.sentiment_analysis, text),
            'keywords': executor.submit(EmotionClassifier.extract_keywords, text)
        }
        print(futures)
        for key, future in futures.items():
            results[key] = future.result()
    return results

if __name__ == "__main__":
    transcribe = AudioTranscriber("output.wav")
    text = transcribe
    results = run_all_parallel(text)
