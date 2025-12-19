import streamlit as st
import torch
import pandas as pd
import sqlite3
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from sarcasm_detector import detect_sarcasm_with_confidence

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="CustomerSense AI",
    page_icon="🧠",
    layout="wide"
)

# =================================================
# DATABASE
# =================================================
DB_PATH = "reviews.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            review_text TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_review(text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO reviews (timestamp, review_text) VALUES (?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text)
    )
    conn.commit()
    conn.close()

def load_reviews():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM reviews", conn)
    conn.close()
    return df

init_db()

# =================================================
# SESSION STATE
# =================================================
if "role" not in st.session_state:
    st.session_state.role = None

if "source_df" not in st.session_state:
    st.session_state.source_df = None

# =================================================
# MODEL
# =================================================
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    )
    model.eval()
    labels = {
        0: "anger",
        1: "disappointment",
        2: "happiness",
        3: "sarcasm"
    }
    return model, tokenizer, labels

model, tokenizer, label_map = load_model()

# =================================================
# LOGIC
# =================================================
emotion_keywords = {
    "anger": ["bad", "rude", "broken", "damaged", "late", "lambat", "teruk", "kasar"],
    "disappointment": ["ok", "not good", "could be better"],
    "happiness": ["good", "nice", "fast", "friendly", "sedap"],
    "sarcasm": ["very nice", "great", "thanks ya"]
}

reason_map = {
    "late": "delivery delay",
    "lambat": "delivery delay",
    "broken": "damaged product",
    "damaged": "damaged product",
    "rude": "staff behaviour issue",
    "kasar": "staff behaviour issue",
    "bad": "poor service quality"
}

# =================================================
# LANDING PAGE
# =================================================
if st.session_state.role is None:
    st.markdown("<h1 style='text-align:center;'>🧠 CustomerSense AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>AI-Based Customer Emotion Intelligence Platform</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧑‍💬 Customer Feedback", use_container_width=True):
            st.session_state.role = "customer"
            st.rerun()

    with col2:
        if st.button("🏢 Business Dashboard", use_container_width=True):
            st.session_state.role = "business"
            st.rerun()

# =================================================
# CUSTOMER PAGE
# =================================================
elif st.session_state.role == "customer":
    st.markdown("## 🧑‍💬 Customer Feedback")

    review = st.text_area(
        "Please share your experience",
        placeholder="Example: delivery lambat but staff friendly"
    )

    if st.button("Submit Feedback"):
        if review.strip():
            save_review(review)
            st.success("THANK YOU FOR YOUR TIME.")
            st.success("SEE YOU AGAIN.")
            st.success("HAVE A NICE DAY 😊")
            st.session_state.role = None
            st.rerun()
        else:
            st.warning("Please enter your feedback.")

# =================================================
# BUSINESS DASHBOARD
# =================================================
elif st.session_state.role == "business":
    st.markdown("## 🏢 Business Emotion Dashboard")

    tab1, tab2, tab3 = st.tabs([
        "📊 Customer Database",
        "✍️ Single Review",
        "📂 Batch / File Upload"
    ])

    # -------- TAB 1 --------
    with tab1:
        df = load_reviews()
        if df.empty:
            st.info("No customer data available.")
        else:
            st.session_state.source_df = df.copy()

    # -------- TAB 2 --------
    with tab2:
        review = st.text_area("Enter a review to test")
        if st.button("Analyze Review"):
            if review.strip():
                st.session_state.source_df = pd.DataFrame([{
                    "review_text": review,
                    "timestamp": datetime.now()
                }])
            else:
                st.warning("Please enter a review.")

    # -------- TAB 3 --------
    with tab3:
        file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
        if file:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                df.columns = ["review_text"]
            else:
                lines = file.read().decode("utf-8").splitlines()
                df = pd.DataFrame(lines, columns=["review_text"])
            df["timestamp"] = datetime.now()
            st.session_state.source_df = df

    # =================================================
    # ANALYSIS
    # =================================================
    if st.session_state.source_df is not None:
        results = []
        trends = []

        for _, row in st.session_state.source_df.iterrows():
            review = row["review_text"]
            date = pd.to_datetime(row["timestamp"]).date()

            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            detected = {label_map[pred.item()]}
            text = review.lower()

             #---- HYBRID SARCASM DETECTION ----
            is_sarcastic = detect_sarcasm_with_confidence(
                review_text=review,
                sentiment_confidence=conf.item()
            )
            
            if is_sarcastic:
                detected.add("sarcasm")

            for emo, keys in emotion_keywords.items():
                for k in keys:
                    if k in text:
                        detected.add(emo)

            reasons = list({v for k, v in reason_map.items() if k in text})

            results.append({
                "Review": review,
                "Emotions": ", ".join(detected),
                "Confidence (%)": round(conf.item() * 100, 2),
                "Reasons": ", ".join(reasons)
            })

            for e in detected:
                trends.append({"date": date, "emotion": e})

        res_df = pd.DataFrame(results)
        trend_df = pd.DataFrame(trends)

        st.markdown("### 📊 KPIs")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", len(res_df))
        c2.metric("Anger Cases", res_df["Emotions"].str.contains("anger").sum())
        c3.metric("Avg Confidence", f"{res_df['Confidence (%)'].mean():.2f}%")

        st.markdown("### 📊 Emotion Distribution")
        st.bar_chart(res_df["Emotions"].str.get_dummies(sep=", ").sum())

        st.markdown("### 📈 Emotion Trend")
        st.line_chart(trend_df.groupby(["date", "emotion"]).size().unstack(fill_value=0))

        st.markdown("### 📄 Detailed Results")
        st.dataframe(res_df)

        csv = res_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Report", csv, "customersense_report.csv", "text/csv")
