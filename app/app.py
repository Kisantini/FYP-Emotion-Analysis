import streamlit as st
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="CustomerSense AI",
    page_icon="🧠",
    layout="wide"
)

# ==============================
# APP STYLE (APP-LIKE UI)
# ==============================
st.markdown("""
<style>
.app-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    border-left: 6px solid #4CAF50;
}
.big-title {
    font-size: 34px;
    font-weight: 700;
}
.sub-title {
    font-size: 18px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    )
    model.eval()

    label_map = {
        0: "anger",
        1: "disappointment",
        2: "happiness",
        3: "sarcasm"
    }
    return model, tokenizer, label_map

model, tokenizer, label_map = load_model()

# ==============================
# BUSINESS RULES
# ==============================
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

# ==============================
# ANALYSIS FUNCTION
# ==============================
def analyze_reviews(df):
    results = []

    for _, row in df.iterrows():
        text = row["review_text"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        primary_emotion = label_map[pred.item()]
        detected = {primary_emotion}

        text_lower = text.lower()
        for e, keys in emotion_keywords.items():
            for k in keys:
                if k in text_lower:
                    detected.add(e)

        reasons = list({v for k, v in reason_map.items() if k in text_lower})
        departments = list({department_map[r] for r in reasons if r in department_map})
        recommendations = list({recommendation_map[r] for r in reasons if r in recommendation_map})

        results.append({
            "timestamp": datetime.now(),
            "review": text,
            "emotions": ", ".join(detected),
            "confidence (%)": round(confidence.item() * 100, 2),
            "reason": ", ".join(reasons) if reasons else "General sentiment",
            "department": ", ".join(departments) if departments else "Management",
            "recommendation": ", ".join(recommendations) if recommendations else "Monitor feedback"
        })

    return pd.DataFrame(results)

# ==============================
# KPI FUNCTION
# ==============================
def show_business_kpis(df):
    st.markdown("## 📊 Business KPIs")

    total = len(df)

    emotion_count = {}
    for emotions in df["emotions"]:
        for e in emotions.split(", "):
            emotion_count[e] = emotion_count.get(e, 0) + 1

    negative = emotion_count.get("anger", 0) + emotion_count.get("disappointment", 0)
    sarcasm = emotion_count.get("sarcasm", 0)
    avg_conf = df["confidence (%)"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", total)
    c2.metric("Negative Reviews", f"{(negative/total)*100:.1f}%")
    c3.metric("Sarcasm Rate", f"{(sarcasm/total)*100:.1f}%")
    c4.metric("Avg Confidence", f"{avg_conf:.1f}%")

    st.markdown("### 📈 Emotion Distribution")
    st.bar_chart(emotion_count)

    st.markdown("### 🏢 Department Impact")
    st.bar_chart(df["department"].value_counts())

    if negative / total > 0.4:
        st.error("⚠️ High customer dissatisfaction detected. Immediate action required.")

# ==============================
# SESSION STATE
# ==============================
if "role" not in st.session_state:
    st.session_state.role = None

# ==============================
# LANDING PAGE
# ==============================
st.markdown("<div class='big-title'>🧠 CustomerSense AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Based Emotional Analysis of Customer Reviews</div>", unsafe_allow_html=True)
st.markdown("---")

if st.session_state.role is None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.markdown("### 🙋 Customer")
        st.write("Submit feedback or review")
        if st.button("Enter as Customer"):
            st.session_state.role = "customer"
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.markdown("### 🏢 Business")
        st.write("Analyze customer emotions & insights")
        if st.button("Enter as Business"):
            st.session_state.role = "business"
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# CUSTOMER FLOW
# ==============================
elif st.session_state.role == "customer":
    st.markdown("## ✍️ Customer Feedback")
    review = st.text_area("Write your feedback")

    if st.button("Submit Feedback"):
        if review.strip():
            st.success("THANK YOU FOR YOUR TIME. SEE YOU AGAIN. HAVE A NICE DAY 😊")
        else:
            st.warning("Please write something.")

# ==============================
# BUSINESS FLOW
# ==============================
elif st.session_state.role == "business":
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Single Review", "Batch / Upload"]
    )

    if page == "Single Review":
        text = st.text_area("Enter customer review")
        if st.button("Analyze Review"):
            if text.strip():
                df = pd.DataFrame([{"review_text": text}])
                st.session_state.analysis_result = analyze_reviews(df)

    elif page == "Batch / Upload":
        file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
        if file:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                df.columns = ["review_text"]
            else:
                lines = file.read().decode("utf-8").splitlines()
                df = pd.DataFrame(lines, columns=["review_text"])

            if st.button("Analyze File"):
                st.session_state.analysis_result = analyze_reviews(df)

    if "analysis_result" in st.session_state:
        st.markdown("## 📌 Emotion Analysis Results")
        for _, row in st.session_state.analysis_result.iterrows():
            st.markdown("<div class='app-card'>", unsafe_allow_html=True)
            st.markdown(f"**📝 Review:** {row['review']}")
            st.markdown(f"**😐 Emotion(s):** {row['emotions']}")
            st.markdown(f"**📊 Confidence:** {row['confidence (%)']}%")
            st.markdown(f"**⚠️ Reason:** {row['reason']}")
            st.markdown(f"**🏢 Department:** {row['department']}")
            st.markdown(f"**✅ Recommendation:** {row['recommendation']}")
            st.markdown("</div>", unsafe_allow_html=True)

        show_business_kpis(st.session_state.analysis_result)
