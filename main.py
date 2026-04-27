import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from langdetect import detect
from googletrans import Translator

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

from nltk.corpus import stopwords

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# -----------------------------
# PREPROCESSING
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(df):
    df['clean_msg'] = df['message'].apply(clean_text)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(df['clean_msg'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }

    ensemble = VotingClassifier(estimators=[
        ('nb', models["Naive Bayes"]),
        ('lr', models["Logistic Regression"]),
        ('rf', models["Random Forest"])
    ], voting='soft')

    ensemble.fit(X_train, y_train)

    return ensemble, vectorizer

# -----------------------------
# SMART FEATURES
# -----------------------------
translator = Translator()
suspicious_words = ["win", "free", "urgent", "click", "offer", "credit", "loan", "upi", "bank"]

def highlight_words(text):
    for word in suspicious_words:
        text = re.sub(f"\\b{word}\\b", f"**:red[{word}]**", text, flags=re.IGNORECASE)
    return text


def detect_links(text):
    return re.findall(r'(https?://\\S+|bit\\.ly/\\S+|tinyurl\\.com/\\S+)', text)


def categorize_message(text):
    text = text.lower()
    if "upi" in text:
        return "UPI Fraud"
    elif "credit" in text:
        return "Credit Card Scam"
    elif "loan" in text:
        return "Loan Scam"
    elif "bank" in text:
        return "Bank Fraud"
    else:
        return "General Spam"


def translate_sms(text, target_lang):
    try:
        detected_lang = detect(text)
        if detected_lang != target_lang:
            return translator.translate(text, dest=target_lang).text
        return text
    except:
        return text

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="SMS Spam Detector", layout="centered")
st.title("📱 SMS Spam Detection System")
st.markdown("### 🚀 ML-based Smart Spam Detector")

# Load + train
with st.spinner("Loading model..."):
    df = load_data()
    model, vectorizer = train_model(df)

# Input
user_input = st.text_area("Enter SMS Message:")

lang_choice = st.selectbox("Translate to:", ["skip", "en", "hi", "or"])

if st.button("Analyze SMS"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        msg = user_input

        if lang_choice != "skip":
            msg = translate_sms(msg, lang_choice)
            st.info(f"Translated Text: {msg}")

        cleaned = clean_text(msg)
        vector = vectorizer.transform([cleaned])

        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0][1]

        if pred == 1:
            st.error("🚨 SPAM DETECTED")
            st.write(f"**Spam Score:** {round(prob*100,2)}%")

            st.markdown("### Highlighted Message")
            st.markdown(highlight_words(msg))

            st.write("**Category:**", categorize_message(msg))

            links = detect_links(msg)
            if links:
                st.warning(f"⚠️ Suspicious Links: {links}")

            st.info("⚠️ Do NOT click links or share personal info.")
        else:
            st.success("✅ SAFE MESSAGE")

# Footer
st.markdown("---")
st.caption("Made for College ML Project 🚀")
