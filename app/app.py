import streamlit as st
import torch
import pandas as pd
import sqlite3
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification

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
            review_text TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_feedback(review_text):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reviews (timestamp, review_text)
        VALUES (?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), review_text))
    conn.commit()
    conn.close()

def load_reviews():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM reviews", conn)
    conn.close()
    return df

init_database()

# =================================================
# SESSION STATE
# =================================================
if "role" not in st.session_state:
    st.session_state.role = None

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
# EMOTION & REASON LOGIC
# =================================================
emotion_keywords = {
    "anger": ["bad", "rude", "broken", "damage", "damaged", "late", "lambat"],
    "disappointment": ["ok", "not good", "could be better"],
    "happiness": ["good", "nice", "fast", "friendly", "sedap"],
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
    review = st.text_area("Please share your experience")

    if st.button("Submit Feedback"):
        if review.strip():
            save_feedback(review)
            st.success("THANK YOU FOR YOUR TIME.")
            st.success("SEE YOU AGAIN.")
            st.success("HAVE A NICE DAY 😊")
            st.stop()
        else:
            st.warning("Please write something.")

# =================================================
# BUSINESS DASHBOARD
# =================================================
if st.session_state.role == "business":
    st.markdown("## 🏢 Business Emotion Dashboard")

    df = load_reviews()

    if df.empty:
        st.info("No customer feedback available yet.")
        st.stop()

    analysis_results = []
    trend_records = []

    for _, row in df.iterrows():
        review = row["review_text"]
        ts = pd.to_datetime(row["timestamp"]).date()

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

        analysis_results.append({
            "Review": review,
            "Emotions": ", ".join(detected),
            "Confidence (%)": round(confidence.item() * 100, 2),
            "Reasons": ", ".join(reasons)
        })

        for emo in detected:
            trend_records.append({"date": ts, "emotion": emo})

    result_df = pd.DataFrame(analysis_results)
    trend_df = pd.DataFrame(trend_records)

    # ================= KPIs =================
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(result_df))
    col2.metric("Anger Cases", result_df["Emotions"].str.contains("anger").sum())
    col3.metric("Avg Confidence (%)", f"{result_df['Confidence (%)'].mean():.2f}")

    # ================= EMOTION DISTRIBUTION =================
    st.markdown("### 📊 Emotion Distribution")
    emotion_counts = result_df["Emotions"].str.get_dummies(sep=", ").sum()
    st.bar_chart(emotion_counts)

    # ================= TREND CHART =================
    st.markdown("### 📈 Emotion Trend Over Time")
    trend_chart = trend_df.groupby(["date", "emotion"]).size().unstack(fill_value=0)
    st.line_chart(trend_chart)

    # ================= ISSUE ANALYSIS =================
    st.markdown("### 🔥 Main Issues Identified")
    issue_counts = result_df["Reasons"].str.get_dummies(sep=", ").sum()
    issue_counts = issue_counts[issue_counts.index != ""]
    if not issue_counts.empty:
        st.bar_chart(issue_counts)
    else:
        st.info("No critical issues detected yet.")

    # ================= AI INSIGHT =================
    st.markdown("### 🧠 AI Business Insight")
    top_emotion = emotion_counts.idxmax()
    top_issue = issue_counts.idxmax() if not issue_counts.empty else "No major issue"

    st.success(
        f"Most customers are expressing **{top_emotion}**. "
        f"The dominant issue identified is **{top_issue}**. "
        "Management attention is recommended."
    )

    # ================= TABLE & EXPORT =================
    st.markdown("### 📄 Detailed Analysis")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Analysis Report",
        csv,
        "customersense_report.csv",
        "text/csv"
    )
