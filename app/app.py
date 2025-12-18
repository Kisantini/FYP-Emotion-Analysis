import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# =================================================
# SESSION STATE (FOR DASHBOARD ANALYTICS)
# =================================================
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

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
# EMOTION KEYWORDS & REASON MAP
# =================================================
emotion_keywords = {
    "anger": ["bad", "rude", "broken", "damage", "damaged", "late", "lambat"],
    "disappointment": ["ok", "not good", "could be better"],
    "happiness": ["good", "nice", "fast", "friendly"],
    "sarcasm": ["very nice", "great", "thanks a lot"]
}

reason_map = {
    "late": "delivery delay",
    "lambat": "delivery delay",
    "slow": "slow service",
    "broken": "damaged product",
    "damage": "damaged product",
    "damaged": "damaged product",
    "rude": "staff behaviour issue",
    "bad service": "poor service quality",
    "bad": "poor product or service quality"
}

# =================================================
# BUSINESS INTELLIGENCE LOGIC
# =================================================
department_map = {
    "delivery delay": "Logistics & Delivery",
    "damaged product": "Product Quality & Packaging",
    "staff behaviour issue": "Customer Service",
    "poor service quality": "Customer Service"
}

recommendation_map = {
    "delivery delay": "Improve delivery tracking and notify customers proactively.",
    "damaged product": "Enhance packaging quality and perform quality checks before delivery.",
    "staff behaviour issue": "Provide customer service training to frontline staff.",
    "poor service quality": "Review service workflow and staff performance."
}

# =================================================
# SIDEBAR NAVIGATION
# =================================================
st.sidebar.title("CustomerSense AI")
st.sidebar.markdown("AI-powered Customer Emotion Intelligence")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🧠 Live Review Analyzer",
        "📊 Insights",
        "💡 Business Recommendations",
        "ℹ️ About"
    ]
)

# =================================================
# DASHBOARD PAGE
# =================================================
if menu == "🏠 Dashboard":
    st.title("CustomerSense AI")
    st.subheader("Understand What Your Customers Really Feel — In Real Time")

    st.markdown("""
    **CustomerSense AI** is a professional AI system that analyzes customer reviews
    in real time and translates emotions into business intelligence.
    """)

    st.markdown("### 🔍 Key Capabilities")
    st.markdown("""
    - Real-time analysis  
    - Mixed-language & slang support  
    - Multi-emotion detection  
    - Sarcasm recognition  
    - Business impact explanation  
    """)

# =================================================
# LIVE REVIEW ANALYZER
# =================================================
elif menu == "🧠 Live Review Analyzer":
    st.title("Live Review Analyzer")

    analysis_mode = st.radio(
        "Select Analysis Mode",
        ["Single Review", "Batch Reviews"]
    )

    if analysis_mode == "Single Review":
        review_input = st.text_area(
            "Enter customer review",
            placeholder="food ok but staff rude gila"
        )
        reviews = [review_input]
    else:
        batch_text = st.text_area(
            "Enter multiple reviews (one per line)",
            placeholder="delivery lambat\nservice very bad lah\nfood ok but staff rude gila"
        )
        reviews = batch_text.split("\n")

    if st.button("Analyze Review"):
        if any(r.strip() for r in reviews):
            for review in reviews:
                if review.strip() == "":
                    continue

                st.markdown("---")
                st.markdown(f"### 📝 Review: *{review}*")

                # ---------------- MODEL PREDICTION ----------------
                inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)

                primary_emotion = label_map[prediction.item()]
                detected_emotions = set([primary_emotion])

                review_lower = review.lower()

                # ---------------- KEYWORD-BASED EMOTION ----------------
                for emotion, keywords in emotion_keywords.items():
                    for word in keywords:
                        if word in review_lower:
                            detected_emotions.add(emotion)

                # ---------------- SARCASM DETECTION ----------------
                positive_triggers = ["very nice", "great", "excellent"]
                negative_triggers = ["broken", "damaged", "bad", "late", "lambat"]

                for p in positive_triggers:
                    if p in review_lower:
                        for n in negative_triggers:
                            if n in review_lower:
                                detected_emotions.add("sarcasm")

                # SAVE FOR DASHBOARD
                st.session_state.analysis_history.extend(list(detected_emotions))

                # ---------------- OUTPUT ----------------
                st.markdown("### 🧠 Emotion Analysis Result")

                emotion_icons = {
                    "anger": "😡",
                    "disappointment": "😕",
                    "happiness": "😊",
                    "sarcasm": "🙃"
                }

                st.success(" | ".join(
                    f"{emotion_icons.get(e,'')} {e.capitalize()}"
                    for e in detected_emotions
                ))

                # CONFIDENCE
                st.markdown("### 📈 Model Confidence")
                st.progress(float(confidence.item()))
                st.write(f"Confidence Score: **{confidence.item() * 100:.2f}%**")

                # REASONS
                st.markdown("### ❓ Why This Happened")
                reasons = set()
                for word, reason in reason_map.items():
                    if word in review_lower:
                        reasons.add(reason)

                if reasons:
                    for r in reasons:
                        st.write(f"- {r}")
                else:
                    st.write("No specific issue detected.")

                # RISK LEVEL
                st.markdown("### 🚦 Business Priority Level")
                if "anger" in detected_emotions and "sarcasm" in detected_emotions:
                    st.error("🔴 Critical Issue – Immediate attention required")
                elif "anger" in detected_emotions:
                    st.warning("🟡 Needs Attention")
                else:
                    st.success("🟢 Normal Feedback")

                # DEPARTMENT
                st.markdown("### 🏢 Affected Department")
                departments = set()
                for r in reasons:
                    if r in department_map:
                        departments.add(department_map[r])

                if departments:
                    for d in departments:
                        st.write(f"- {d}")
                else:
                    st.write("General feedback")

                # RECOMMENDATION
                st.markdown("### 💡 Recommended Business Action")
                recs = set()
                for r in reasons:
                    if r in recommendation_map:
                        recs.add(recommendation_map[r])

                if recs:
                    for rec in recs:
                        st.write(f"- {rec}")
                else:
                    st.write("No immediate action required.")
        else:
            st.warning("Please enter at least one review.")

# =================================================
# INSIGHTS DASHBOARD
# =================================================
elif menu == "📊 Insights":
    st.title("Customer Emotion Analytics")

    if len(st.session_state.analysis_history) == 0:
        st.info("No analysis data available yet.")
    else:
        emotion_counts = {}
        for e in st.session_state.analysis_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        st.markdown("### 📊 Emotion Distribution")
        st.bar_chart(emotion_counts)

        dominant = max(emotion_counts, key=emotion_counts.get)
        st.markdown("### 📌 Key Insight")
        st.write(f"Most frequent emotion detected: **{dominant.capitalize()}**")

# =================================================
# BUSINESS RECOMMENDATIONS PAGE
# =================================================
elif menu == "💡 Business Recommendations":
    st.title("AI-Driven Business Recommendations")

    st.markdown("""
    This system converts emotional signals into actionable insights
    to support operational and strategic decision-making.
    """)

# =================================================
# ABOUT PAGE
# =================================================
elif menu == "ℹ️ About":
    st.title("About CustomerSense AI")

    st.markdown("""
    CustomerSense AI is a BERT-based real-time emotion analysis system
    developed as a Final Year Project (FYP).

    It supports mixed-language reviews, sarcasm detection,
    explainable AI output, and business intelligence features.
    """)
