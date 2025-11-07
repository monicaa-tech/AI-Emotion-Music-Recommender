# AI-Emotion-Music-Recommender
# An AI-powered chatbot that detects user emotions and recommends songs based on mood using NLP and Spotify dataset.
THIS IS MUSIC RECOMMENDATION 
# ✅ Step 1: Import Libraries
import pandas as pd
import torch
from transformers import pipeline
import gradio as gr

# ✅ Step 2: Load a lightweight Emotion Detection Model
emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# ✅ Step 3: Define Mood-to-Music Mapping
mood_to_music = {
    "joy": ["Happy – Pharrell Williams", "Good as Hell – Lizzo", "Can't Stop the Feeling – Justin Timberlake"],
    "sadness": ["Someone Like You – Adele", "Let Her Go – Passenger", "Fix You – Coldplay"],
    "anger": ["Believer – Imagine Dragons", "Numb – Linkin Park", "Lose Yourself – Eminem"],
    "fear": ["Lovely – Billie Eilish", "The Night We Met – Lord Huron", "Creep – Radiohead"],
    "love": ["Perfect – Ed Sheeran", "All of Me – John Legend", "Just the Way You Are – Bruno Mars"],
    "surprise": ["Happy Now – Kygo", "Wake Me Up – Avicii", "Counting Stars – OneRepublic"],
    "neutral": ["Let It Be – The Beatles", "Photograph – Ed Sheeran", "Best Day of My Life – American Authors"]
}

# ✅ Step 4: Chatbot Logic
def chatbot_response(message, chat_history=[]):
    # Analyze Emotion
    emotion = emotion_analyzer(message)[0]['label'].lower()
    
  # Pick Songs for that Emotion
  songs = mood_to_music.get(emotion, mood_to_music["neutral"])
    
   # Create Response
   response = f"🎭 I sense you're feeling **{emotion}**.\nHere are some songs that might match your mood:\n"
    for s in songs:
        response += f"🎵 {s}\n"
    
   chat_history.append((message, response))
    return "", chat_history

# ✅ Step 5: Gradio Chatbot UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎧 Welcome to Emotion-Based AI Music Recommender 🎶")
    gr.Markdown("Hey there! 👋 Tell me how you're feeling, and I’ll suggest songs to match your mood.")
    
  chatbot = gr.Chatbot(label="AI Mood Assistant 🤖")
    msg = gr.Textbox(placeholder="Type how you feel...", label="Your Mood 💬")
    clear = gr.Button("Clear Chat")

   msg.submit(chatbot_response, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
