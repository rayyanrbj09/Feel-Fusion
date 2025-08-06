import os 
import sys
import google.generativeai as genai


# Add parent folder to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from emotion_pipeline import EmotionClassifier

# Configuration
secret_key = os.environ.get('SECRET_KEY', 'your_secret_key') 


api_key = 'AIzaSyDoKZnfigtf1CWTTHFZts9EkTLrtvEANsA'

# Initialize Gemini model
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

def get_activity_suggestions(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return "Error: Could not generate suggestions."

# Example usage
if __name__ == "__main__":
    emotion_classifier = EmotionClassifier()

    
    # Get emotion or sentiment dictionary if needed
    emotion_data = emotion_classifier.save_to_dictionary(emotion_classifier)
    
    prompt = f"""
    The user is feeling {emotion_data.get('emotion', 'happy')} and has a sentiment of {emotion_data.get('sentiment', 'positive')}.
    Suggest 5 activities that can help the user feel better.
    The activities should be diverse and suitable for different preferences.
    Be creative, specific, and provide a brief description of each activity, Add a fun emoji to each activity.
    """
    
    suggestions = get_activity_suggestions(prompt)
    print("Activity Suggestions:\n", suggestions)