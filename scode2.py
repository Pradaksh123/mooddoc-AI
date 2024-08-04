import streamlit as st
from transformers import pipeline
import PyPDF2
import matplotlib.pyplot as plt
import numpy as np

# Load the sentiment analysis model from Hugging Face
sentiment_model = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

# Function to read PDF and extract text
def read_pdf(file):
    pdf_text = ''
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() + ' '  # Extract text from each page
    return pdf_text

# Function to read text file and extract text
def read_text_file(file):
    text_content = file.read().decode("utf-8")
    return text_content

# Function to analyze sentiment of the text
def analyze_sentiment(text):
    results = sentiment_model(text)
    return results

# Streamlit interface
st.title("🌈 MoodDoc, Sentiment Analysis Tool For Files and Docs 📄")
# Sidebar settings
st.sidebar.header("Settings")
st.sidebar.markdown("### Model Description")
st.sidebar.write("""
- **Model Name**: BERT-based Emotion Analysis Model
- **Type**: Transformer-based model for text classification
- **Factors**: The model parameters are based on:
  - Pre-trained on a large corpus of text data
  - Fine-tuned for emotion detection in text
  - Utilizes attention mechanisms to understand context and sentiment
""")

# File uploader for both text and PDF files
uploaded_file = st.file_uploader("Choose a text or PDF file", type=["txt", "pdf"])

if uploaded_file is not None:
    # Check the file type and read accordingly
    if uploaded_file.type == "application/pdf":
        text_content = read_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        text_content = read_text_file(uploaded_file)

    # Display the content of the uploaded file in a text area
    st.subheader("Uploaded File Content:")
    st.text_area("Content", text_content, height=300)
    # Footer
    st.markdown("<footer style='text-align: center;'>Content is displayed above</footer>", unsafe_allow_html=True)

    # Analyze the sentiment of the extracted text
    sentiment_results = analyze_sentiment(text_content)

    # Display model predictions for each sentence
    st.subheader("Model Predictions for Each Sentence:")
    sentences = text_content.split('. ')
    for sentence in sentences:
        if sentence.strip():  # Check if the sentence is not empty
            result = sentiment_model(sentence)
            st.write(f"Sentence: **{sentence}**")
            st.write(f"Predicted Emotion: **{result[0]['label']}** with confidence: **{result[0]['score']:.7f}**")
            st.write("---")

    # Initialize scores for all six emotions
    emotion_scores = {
        'sadness': 0.0,
        'joy': 0.0,
        'anger': 0.0,
        'fear': 0.0,
        'surprise': 0.0,
        'neutral': 0.0
    }

    # Calculate scores based on results
    for result in sentiment_results:
        emotion_label = result['label']
        emotion_score = result['score'] * 100  # Scale score to 1-100
        if emotion_label in emotion_scores:
            emotion_scores[emotion_label] += emotion_score

    # Ensure scores are non-negative and scale them to 1-100
    scores = [max(emotion_scores[emotion], 0) for emotion in ['sadness', 'joy', 'anger', 'fear', 'surprise', 'neutral']]

    # Define labels and emojis for each emotion
    labels = ['Sadness ', 'Joy ', 'Anger ', 'Fear ', 'Surprise ', 'Neutral ']

    # Display numerical scores with bold formatting and emojis
    st.subheader("Numerical Scores:")
    for emotion, score, emoji in zip(emotion_scores.keys(), scores, ['😢', '😄', '😠', '😨', '😮', '😐']):
        st.write(f"**{emotion.capitalize()} {emoji}**: {score:.7f}")

    # Display the sentiment analysis results
    st.subheader("Sentiment Analysis Results")

    # Bar Graph
    st.subheader("Bar Graph of Emotion Scores")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, scores, color=['#FF6B6B', '#FFD93D', '#69B578', '#FF8C00', '#4682B4', '#D3D3D3'])
    ax.set_ylim(0, 100)  # Set y-axis limit from 0 to 100
    ax.set_ylabel('Scores (1-100)')
    ax.set_title('Emotion Scores')
    st.pyplot(fig)

    # Footer
    st.markdown("<footer style='text-align: center;'>Content is displayed above</footer>", unsafe_allow_html=True)
