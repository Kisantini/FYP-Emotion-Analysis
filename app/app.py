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
    cursor.execute(
        "INSERT INTO reviews (timestamp, review_text) VALUES (?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), review_text)
    )
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

if st.session_state.role == "business":
    st.markdown("## 🏢 Business Emotion Dashboard")

    tab1, tab2, tab3 = st.tabs([
        "📊 Customer Database",
        "✍️ Single Review",
        "📂 Batch / File Upload"
    ])

    # =================================================
    # TAB 1 — CUSTOMER DATABASE ANALYSIS
    # =================================================
    with tab1:
        st.markdown("### 📊 Analysis from Real Customer Feedback")

        df = load_reviews()

        if df.empty:
            st.info("No customer feedback available yet.")
            st.stop()

        source_df = df.copy()

    # =================================================
    # TAB 2 — SINGLE REVIEW ANALYSIS
    # =================================================
    with tab2:
        st.markdown("### ✍️ Single Review Testing")

        review = st.text_area(
            "Enter a review for testing",
            placeholder="Example: delivery lambat but staff rude gila"
        )

        if st.button("Analyze Single Review"):
            if review.strip():
                source_df = pd.DataFrame([{
                    "review_text": review,
                    "timestamp": datetime.now()
                }])
            else:
                st.warning("Please enter a review.")
                st.stop()

    # =================================================
    # TAB 3 — BATCH / FILE UPLOAD ANALYSIS
    # =================================================
    with tab3:
        st.markdown("### 📂 Batch / File Upload Analysis")

        file = st.file_uploader(
            "Upload CSV or TXT file",
            type=["csv", "txt"]
        )

        if file:
            if file.name.endswith(".csv"):
                temp_df = pd.read_csv(file)
                temp_df.columns = ["review_text"]
                source_df = temp_df
                source_df["timestamp"] = datetime.now()
            else:
                lines = file.read().decode("utf-8").splitlines()
                source_df = pd.DataFrame(lines, columns=["review_text"])
                source_df["timestamp"] = datetime.now()

    # =================================================
    # ANALYSIS ENGINE (COMMON FOR ALL TABS)
    # =================================================
    if "source_df" in locals():

        analysis_results = []
        trend_data = []

        for _, row in source_df.iterrows():
            review = row["review_text"]
            date = pd.to_datetime(row["timestamp"]).date()

            inputs = tokenizer(
                review,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)

            detected = {label_map[pred.item()]}
            review_lower = review.lower()

            for emo, keys in emotion_keywords.items():
                for k in keys:
                    if k in review_lower:
                        detected.add(emo)

            reasons = list({v for k, v in reason_map.items() if k in review_lower})

            analysis_results.append({
                "Review": review,
                "Emotions": ", ".join(detected),
                "Confidence (%)": round(confidence.item() * 100, 2),
                "Reasons": ", ".join(reasons)
            })

            for emo in detected:
                trend_data.append({"date": date, "emotion": emo})

        result_df = pd.DataFrame(analysis_results)
        trend_df = pd.DataFrame(trend_data)

        # =================================================
        # KPI METRICS
        # =================================================
        st.markdown("### 📊 Key Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", len(result_df))
        col2.metric("Anger Cases", result_df["Emotions"].str.contains("anger").sum())
        col3.metric("Avg Confidence (%)", f"{result_df['Confidence (%)'].mean():.2f}")

        # =================================================
        # EMOTION DISTRIBUTION
        # =================================================
        st.markdown("### 📊 Emotion Distribution")
        emotion_counts = result_df["Emotions"].str.get_dummies(sep=", ").sum()
        st.bar_chart(emotion_counts)

        # =================================================
        # EMOTION TREND
        # =================================================
        st.markdown("### 📈 Emotion Trend Over Time")
        trend_chart = trend_df.groupby(["date", "emotion"]).size().unstack(fill_value=0)
        st.line_chart(trend_chart)

        # =================================================
        # ISSUE ANALYSIS
        # =================================================
        st.markdown("### 🔥 Main Issues Identified")
        issue_counts = result_df["Reasons"].str.get_dummies(sep=", ").sum()
        issue_counts = issue_counts[issue_counts.index != ""]

        if not issue_counts.empty:
            st.bar_chart(issue_counts)
        else:
            st.info("No major issues detected.")

        # =================================================
        # AI INSIGHT
        # =================================================
        st.markdown("### 🧠 AI Business Insight")

        top_emotion = emotion_counts.idxmax()
        top_issue = issue_counts.idxmax() if not issue_counts.empty else "No dominant issue"

        st.success(
            f"Customers mostly express **{top_emotion}** emotion. "
            f"The most common issue is **{top_issue}**. "
            "Immediate action is recommended."
        )

        # =================================================
        # TABLE & EXPORT
        # =================================================
        st.markdown("### 📄 Detailed Analysis")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Analysis Report",
            csv,
            "customersense_report.csv",
            "text/csv"
        )
