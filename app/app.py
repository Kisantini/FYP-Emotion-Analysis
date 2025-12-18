import streamlit as st
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import sqlite3
from datetime import datetime

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

# =================================================
# SAVE CUSTOMER FEEDBACK
# =================================================
def save_feedback_to_db(review_text):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reviews (timestamp, review_text, emotions, confidence, reasons)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        review_text,
        None,
        None,
        None
    ))

    conn.commit()
    conn.close()
    
# =================================================
# APP CONFIG
# =================================================
st.set_page_config(
    page_title="CustomerSense AI",
    page_icon="🧠",
    layout="wide"
)

# =================================================
# SESSION STATE
# =================================================
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

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
    return model, tokenizer, label_map


model, tokenizer, label_map = load_model()
model.eval()

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

department_map = {
    "delivery delay": "Logistics",
    "damaged product": "Quality Control",
    "staff behaviour issue": "Customer Service",
    "poor service quality": "Customer Service"
}

recommendation_map = {
    "delivery delay": "Improve delivery tracking and communication.",
    "damaged product": "Improve packaging and quality inspection.",
    "staff behaviour issue": "Conduct customer service training.",
    "poor service quality": "Review service workflow."
}

# =================================================
# HEADER
# =================================================
st.markdown(
    """
    <h1 style='text-align:center;'>🧠 CustomerSense AI</h1>
    <p style='text-align:center;'>Customer Emotion Intelligence Platform</p>
    """,
    unsafe_allow_html=True
)

# =================================================
# TABS
# =================================================
tab1, tab2, tab3 = st.tabs(["🧠 Analyze", "📊 Dashboard", "📥 Reports & KPIs"])

# =================================================
# TAB 1 — ANALYZE
# =================================================
with tab1:
    st.markdown("## 🔍 Step 1: Choose Input Method")

    input_mode = st.selectbox(
        "Select review input type",
        ["✍️ Single Review", "📋 Batch Text", "📂 Upload File"]
    )

    reviews = []

    if input_mode == "✍️ Single Review":
        text = st.text_area("Enter customer review")
        if text.strip():
            reviews = [text]

    elif input_mode == "📋 Batch Text":
        batch = st.text_area("Enter multiple reviews (one per line)")
        reviews = [r for r in batch.split("\n") if r.strip()]

    else:
        file = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"])
        if file:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                reviews = df.iloc[:, 0].dropna().tolist()
            else:
                reviews = file.read().decode("utf-8").splitlines()

    if st.button("🚀 Analyze"):
        if not reviews:
            st.warning("Please provide at least one review.")
        else:
            st.markdown("---")
            st.markdown("## 📄 Analysis Results")

            for review in reviews:
                st.markdown(f"### 📝 Review: *{review}*")

                inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                confidence, pred = torch.max(probs, dim=1)

                primary_emotion = label_map[pred.item()]
                detected = set([primary_emotion])

                review_lower = review.lower()
                for e, keys in emotion_keywords.items():
                    for k in keys:
                        if k in review_lower:
                            detected.add(e)

                reasons = list({v for k, v in reason_map.items() if k in review_lower})

                # Save history
                st.session_state.analysis_history.extend(detected)

                record = {
                    "timestamp": datetime.now(),
                    "review": review,
                    "emotions": ", ".join(detected),
                    "confidence": round(confidence.item() * 100, 2),
                    "reasons": ", ".join(reasons)
                }
                st.session_state.analysis_records.append(record)

                st.success(" | ".join(detected))
                st.progress(float(confidence.item()))
                st.caption(f"Confidence: {confidence.item()*100:.2f}%")

                if reasons:
                    st.write("**Reason:**", ", ".join(reasons))

                st.markdown("---")

# =================================================
# TAB 2 — DASHBOARD (ADMIN KPI VIEW)
# =================================================
with tab2:
    st.markdown("## 📊 Admin Analytics Dashboard")

    if not st.session_state.analysis_history:
        st.info("No data available yet.")
    else:
        df = pd.DataFrame(st.session_state.analysis_records)

        # KPI METRICS
        total_reviews = len(df)
        avg_conf = df["confidence"].mean()

        emotion_count = {}
        for e in st.session_state.analysis_history:
            emotion_count[e] = emotion_count.get(e, 0) + 1

        critical_cases = emotion_count.get("anger", 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", total_reviews)
        col2.metric("Avg Confidence (%)", f"{avg_conf:.2f}")
        col3.metric("Anger Cases", critical_cases)

        st.markdown("### 📊 Emotion Distribution")
        st.bar_chart(emotion_count)

# =================================================
# TAB 3 — EXPORT REPORTS
# =================================================
with tab3:
    st.markdown("## 📥 Export Analysis Reports")

    if not st.session_state.analysis_records:
        st.info("No analysis data to export.")
    else:
        export_df = pd.DataFrame(st.session_state.analysis_records)

        st.dataframe(export_df)

        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv,
            file_name="customersense_report.csv",
            mime="text/csv"
        )

        st.success("Report ready for business use.")
