import streamlit as st
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# CSS Styling
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
}
.report-box {
    background: rgba(255,255,255,0.12);
    padding: 18px;
    border-radius: 15px;
    margin-top: 15px;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.1);
}
.badge {
    padding: 8px 15px;
    border-radius: 20px;
    display: inline-block;
    margin: 6px 0;
    font-weight: bold;
    color: white;
}
.cat {background-color:#008cff;}
.pos {background-color:#00c957;}
.neg {background-color:#ff4747;}
.neu {background-color:#ffbf00;}
.status-green {color:#00ff85; font-weight:bold;}
.status-yellow {color:#ffd000; font-weight:bold;}
.status-blue {color:#57c1ff; font-weight:bold;}
.status-red {color:#ff4c4c; font-weight:bold;}
</style>
""", unsafe_allow_html=True)


# Load Models
hf_token = "################"
nlp = spacy.load("en_core_web_sm")
news_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-mnli", use_auth_token=hf_token)
sentiment = SentimentIntensityAnalyzer()

candidate_labels = [
    "technology news","sports news","political news","crime report","entertainment news",
    "financial/business news","health and medicine","science discovery","travel news","breaking news",
    "disaster news", "environmental news", "weather news"
]

# Load Baseline Data for Anomaly Detection
BASELINE = r"D:\Anomaly_hypertext_news_detection\Preprocessing\cleaned_final_dataset.csv"
df = pd.read_csv(BASELINE)
text_col = "cleaned_text" if "cleaned_text" in df.columns else df.columns[0]
baseline_data = df[text_col].astype(str).tolist()

vec = TfidfVectorizer(stop_words="english")
baseline_vectors = vec.fit_transform(baseline_data)


def detect_location(text):
    doc = nlp(text)
    locs = [ent.text for ent in doc.ents if ent.label_ in ["GPE","LOC"]]
    if not locs:
        pattern = r"\b[A-Z][a-z]+(?: city| district| state)?\b"
        locs = re.findall(pattern, text)
    return locs if locs else ["Not Detected"]


def anomaly_score(text, confidence):
    val = vec.transform([text])
    sim = cosine_similarity(val, baseline_vectors).mean()
    score = min(1, (1 - sim) + (1 - confidence) * 0.2)
    return score, sim


# ---------- UI Section ----------
st.title("News Anomaly Detection")
st.write("Analyze tone, category, anomalies, and locations with full visual representation.")

article = st.text_area("📝 Paste a News Article:")
date = st.date_input("📅 Select Publication Date")

if st.button("Analyze News"):
    if not article.strip():
        st.error("⚠ Please enter an article.")
    else:
        res = news_classifier(article, candidate_labels)
        category, conf = res['labels'][0], round(res['scores'][0], 3)
        
        sent = sentiment.polarity_scores(article)["compound"]
        sent_type = "Positive" if sent >= 0.05 else "Negative" if sent <= -0.05 else "Neutral"
        
        anomaly, similarity = anomaly_score(article, conf)
        locations = detect_location(article)

        # ---------- Output Section ----------
        st.markdown("<div class='report-box'>", unsafe_allow_html=True)
        
        st.markdown(f"### 📰 Article Summary")
        st.write(article)
        st.write(f"📅 **Publication Date:** `{date}`")

        # Category Badge
        st.markdown(f"<span class='badge cat'>📌 Category: {category} ({conf})</span>", unsafe_allow_html=True)

        # Sentiment Badge
        color = "pos" if sent_type=="Positive" else "neg" if sent_type=="Negative" else "neu"
        st.markdown(f"<span class='badge {color}'>💬 Sentiment: {sent_type}</span>", unsafe_allow_html=True)

        # Location
        st.write(f"📍 **Detected Locations:** `{', '.join(locations)}`")

        # Score Indicators
        st.write(f"🔍 Similarity Score: **{round(similarity,3)}**")
        st.write(f"🚨 Anomaly Score: **{round(anomaly,3)}**")

        # Status Color Logic
        if anomaly <= 0.3:
            status = "<p class='status-green'>🟢 Normal Verified News</p>"
        elif anomaly <= 0.6:
            status = "<p class='status-yellow'>🟡 Rare — Needs Review</p>"
        elif anomaly <= 0.85:
            status = "<p class='status-blue'>🔵 Unusual — Low Familiarity</p>"
        else:
            status = "<p class='status-red'>🔴 Highly Deviating — Check Source</p>"

        st.markdown(status, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# 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