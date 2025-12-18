import streamlit as st
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import sqlite3
from datetime import datetime

# =================================================
# APP CONFIG
# =================================================
st.set_page_config(
    page_title="CustomerSense AI",
    page_icon="🧠",
    layout="wide"
)

# =================================================
# DATABASE SETUP
# =================================================
DB_PATH = "reviews.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            review_text TEXT,
            emotions TEXT,
            confidence REAL,
            reasons TEXT
        )
    """)
    conn.commit()
    conn.close()

init_database()

def save_feedback_to_db(review_text):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reviews (timestamp, review_text)
        VALUES (?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        review_text
    ))
    conn.commit()
    conn.close()

def load_reviews_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM reviews", conn)
    conn.close()
    return df

# =================================================
# SESSION STATE
# =================================================
if "role" not in st.session_state:
    st.session_state.role = None

if "analysis_records" not in st.session_state:
    st.session_state.analysis_records = []

# =================================================
# LOAD MODEL
# =================================================
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    )
    label_map = {
        0: "anger",
        1: "disappointment",
        2: "happiness",
        3: "sarcasm"
    }
    model.eval()
    return model, tokenizer, label_map

model, tokenizer, label_map = load_model()

# =================================================
# BUSINESS LOGIC
# =================================================
emotion_keywords = {
    "anger": ["bad", "rude", "broken", "damage", "damaged", "late", "lambat"],
    "disappointment": ["ok", "not good", "could be better"],
    "happiness": ["good", "nice", "fast", "friendly"],
    "sarcasm": ["very nice", "great"]
}

reason_map = {
    "late": "delivery delay",
    "lambat": "delivery delay",
    "broken": "damaged product",
    "damaged": "damaged product",
    "rude": "staff behaviour issue",
    "bad": "poor service quality"
}

# =================================================
# LANDING PAGE
# =================================================
if st.session_state.role is None:
    st.markdown("<h1 style='text-align:center;'>🧠 CustomerSense AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Customer Emotion Intelligence Platform</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧑‍💬 Customer Feedback", use_container_width=True):
            st.session_state.role = "customer"
            st.rerun()

    with col2:
        if st.button("🏢 Business Dashboard", use_container_width=True):
            st.session_state.role = "business"
            st.rerun()

    st.stop()

# =================================================
# CUSTOMER PAGE
# =================================================
if st.session_state.role == "customer":
    st.markdown("## 🧑‍💬 Customer Feedback")
    feedback = st.text_area(
        "Please share your experience",
        placeholder="Type your review here..."
    )

    if st.button("Submit Feedback"):
        if feedback.strip():
            save_feedback_to_db(feedback)
            st.success("THANK YOU FOR YOUR TIME.")
            st.success("SEE YOU AGAIN.")
            st.success("HAVE A NICE DAY 😊")
            st.stop()
        else:
            st.warning("Please enter your feedback.")

# =================================================
# BUSINESS DASHBOARD
# =================================================
if st.session_state.role == "business":
    st.markdown("## 🏢 Business Emotion Dashboard")

    df = load_reviews_from_db()

    if df.empty:
        st.info("No customer feedback available yet.")
        st.stop()

    reviews = df["review_text"].dropna().tolist()
    results = []

    for review in reviews:
        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        detected = {label_map[pred.item()]}
        review_lower = review.lower()

        for e, keys in emotion_keywords.items():
            for k in keys:
                if k in review_lower:
                    detected.add(e)

        reasons = list({v for k, v in reason_map.items() if k in review_lower})

        results.append({
            "Review": review,
            "Emotions": ", ".join(detected),
            "Confidence (%)": round(confidence.item() * 100, 2),
            "Reasons": ", ".join(reasons)
        })

    result_df = pd.DataFrame(results)

    # KPIs
    st.markdown("### 📊 Key Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Total Reviews", len(result_df))
    col2.metric("Anger Cases", result_df["Emotions"].str.contains("anger").sum())

    st.markdown("### 📄 Detailed Analysis")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Analysis Report",
        csv,
        "customersense_report.csv",
        "text/csv"
    )
