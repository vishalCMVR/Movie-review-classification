import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
page_title="SentimentAI",
page_icon="🎬",
layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown('''

<style>
.main {
    background-color: #0E1117;
}

.hero {
    text-align: center;
    padding: 2rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #B0B3B8;
}

.result-box {
    padding: 1.5rem;
    border-radius: 15px;
    background: #1A1D24;
    border: 1px solid #2E3138;
}

.metric-card {
    background: #1A1D24;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
}
</style>'''

, unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_sentiment_model():
    return load_model("review_model.h5")

model = load_sentiment_model()

# ---------------- IMDB WORD INDEX ----------------

word_index = imdb.get_word_index()

MAX_FEATURES = 10000
MAX_LEN = 500

# ---------------- PREPROCESS ----------------

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()

    encoded_review = []

    for word in words:
        idx = word_index.get(word, 2) + 3

        if idx >= MAX_FEATURES:
            idx = 2

        encoded_review.append(idx)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN
    )

    return padded_review

# ---------------- PREDICTION ----------------

def predict_sentiment(review):

    processed = preprocess_text(review)

    prediction = model.predict(
        processed,
        verbose=0
    )

    score = float(prediction[0][0])

    sentiment = (
        "Positive 😊"
        if score > 0.5
        else "Negative 😞"
    )

    return sentiment, score
    

# ---------------- HERO SECTION ----------------

st.markdown("""

<style>

.hero-container{
    text-align:center;
    padding:60px 20px;
    border-radius:25px;
    background: linear-gradient(
        135deg,
        rgba(59,130,246,0.15),
        rgba(139,92,246,0.15)
    );
    backdrop-filter: blur(20px);
    border:1px solid rgba(255,255,255,0.1);
    margin-bottom:30px;
}

.hero-badge{
    display:inline-block;
    padding:8px 16px;
    border-radius:50px;
    background:#111827;
    color:#60A5FA;
    font-size:14px;
    font-weight:600;
    margin-bottom:20px;
}

.hero-title{
    font-size:64px;
    font-weight:900;
    line-height:1;
    background:linear-gradient(
        90deg,
        #60A5FA,
        #8B5CF6,
        #EC4899
    );
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero-subtitle{
    margin-top:20px;
    font-size:22px;
    color:#9CA3AF;
    max-width:850px;
    margin-left:auto;
    margin-right:auto;
    line-height:1.8;
}

.hero-stats{
    margin-top:35px;
    display:flex;
    justify-content:center;
    gap:50px;
}

.stat-number{
    font-size:28px;
    font-weight:700;
    color:white;
}

.stat-label{
    color:#9CA3AF;
}

</style>

<div class="hero-container">

<div class="hero-badge">
🚀 Deep Learning Powered NLP System
</div>

<div class="hero-title">
SentimentAI
</div>

<div class="hero-subtitle">
Transform movie reviews into actionable insights using a
SimpleRNN-based Deep Learning model trained on the IMDB
dataset. Instantly detect sentiment with AI-powered
text understanding.
</div>

<div class="hero-stats">

<div>
<div class="stat-number">78%</div>
<div class="stat-label">Accuracy</div>
</div>

<div>
<div class="stat-number">25K+</div>
<div class="stat-label">Reviews Trained</div>
</div>

<div>
<div class="stat-number">RNN</div>
<div class="stat-label">Architecture</div>
</div>

</div>

</div>
""", unsafe_allow_html=True)


# ---------------- INPUT ----------------

review = st.text_area(
"Enter Movie Review",
height=200,
placeholder="Example: This movie was amazing and I loved every minute of it..."
)

# ---------------- BUTTON ----------------

if st.button("🚀 Analyze Sentiment", use_container_width=True):

    if review.strip():

        sentiment, score = predict_sentiment(review)

        confidence = (
            score * 100
            if score > 0.5
            else (1 - score) * 100
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted Sentiment",
                value=sentiment
            )

        with col2:
            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

        st.progress(float(confidence / 100))

        st.markdown(
            f"""
            <div class="result-box">
                <h3>Prediction Score</h3>
                <p style="font-size:24px;">
                    {score:.4f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("Please enter a movie review.")

# ---------------- FOOTER ----------------

st.markdown("---")
st.caption(
"Built with TensorFlow, Keras and Streamlit"
)
