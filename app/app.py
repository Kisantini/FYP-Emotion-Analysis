import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

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
    **CustomerSense AI** is a professional AI system designed to help businesses
    understand customer emotions beyond simple positive or negative labels.
    """)

    st.markdown("### 🔍 Key Capabilities")
    st.markdown("""
    - Real-time customer review analysis  
    - Mixed-language & slang understanding  
    - Multi-emotion detection  
    - Sarcasm recognition  
    - Explainable business insights  
    """)

    st.info("Designed for business owners, managers, and customer support teams.")

# =================================================
# LIVE REVIEW ANALYZER
# =================================================
elif menu == "🧠 Live Review Analyzer":
    st.title("Live Review Analyzer")
    st.markdown("Analyze customer feedback instantly and receive actionable insights.")

    review = st.text_area(
        "Enter customer review",
        placeholder="Example: food ok but staff rude gila"
    )

    if st.button("Analyze Review"):
        if review.strip() != "":
            # ----------- MODEL PREDICTION -----------
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1)

            primary_emotion = label_map[prediction.item()]
            detected_emotions = set()
            detected_emotions.add(primary_emotion)

            review_lower = review.lower()

            # ----------- KEYWORD EMOTION DETECTION -----------
            for emotion, keywords in emotion_keywords.items():
                for word in keywords:
                    if word in review_lower:
                        detected_emotions.add(emotion)

            # ----------- SARCASM DETECTION -----------
            positive_triggers = ["very nice", "great", "excellent"]
            negative_triggers = ["broken", "damaged", "bad", "late", "lambat"]

            for p in positive_triggers:
                if p in review_lower:
                    for n in negative_triggers:
                        if n in review_lower:
                            detected_emotions.add("sarcasm")

            # =================================================
            # OUTPUT SECTION (BUSINESS-GRADE)
            # =================================================
            st.markdown("### 🧠 Emotion Analysis Result")

            emotion_icons = {
                "anger": "😡",
                "disappointment": "😕",
                "happiness": "😊",
                "sarcasm": "🙃"
            }

            emotion_display = [
                f"{emotion_icons.get(e, '')} {e.capitalize()}"
                for e in detected_emotions
            ]

            st.success(" | ".join(emotion_display))

            # ----------- REASONS -----------
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

            # ----------- RISK LEVEL -----------
            st.markdown("### 🚦 Business Priority Level")

            if "anger" in detected_emotions and "sarcasm" in detected_emotions:
                st.error("🔴 Critical Issue – Immediate attention required")
            elif "anger" in detected_emotions:
                st.warning("🟡 Needs Attention")
            else:
                st.success("🟢 Normal Feedback")

            # ----------- DEPARTMENT IMPACT -----------
            st.markdown("### 🏢 Affected Department")

            affected_departments = set()
            for r in reasons:
                if r in department_map:
                    affected_departments.add(department_map[r])

            if affected_departments:
                for d in affected_departments:
                    st.write(f"- {d}")
            else:
                st.write("General feedback")

            # ----------- BUSINESS RECOMMENDATION -----------
            st.markdown("### 💡 Recommended Business Action")

            recommendations = set()
            for r in reasons:
                if r in recommendation_map:
                    recommendations.add(recommendation_map[r])

            if recommendations:
                for rec in recommendations:
                    st.write(f"- {rec}")
            else:
                st.write("No immediate action required.")

        else:
            st.warning("Please enter a review")

# =================================================
# INSIGHTS PAGE
# =================================================
elif menu == "📊 Insights":
    st.title("Customer Insights Overview")
    st.markdown("""
    This section summarizes emotional trends from analyzed customer reviews.
    Advanced charts and analytics can be added in future versions.
    """)

    st.info("Analyze more customer reviews to generate insights.")

# =================================================
# BUSINESS RECOMMENDATIONS PAGE
# =================================================
elif menu == "💡 Business Recommendations":
    st.title("AI-Driven Business Recommendations")

    st.markdown("""
    CustomerSense AI transforms emotional feedback into actionable business insights
    to support management decision-making.
    """)

    st.markdown("""
    **Examples:**
    - Delivery delays → Improve logistics tracking  
    - Staff complaints → Customer service training  
    - Product damage → Better packaging and quality control  
    """)

    st.success("Designed to help businesses respond proactively to customer issues.")

# =================================================
# ABOUT PAGE
# =================================================
elif menu == "ℹ️ About":
    st.title("About CustomerSense AI")

    st.markdown("""
    CustomerSense AI is a BERT-based emotion analysis system developed as a
    Final Year Project (FYP).

    The system supports:
    - Mixed-language customer reviews  
    - Multi-emotion detection  
    - Sarcasm understanding  
    - Explainable, business-focused outputs  

    This project focuses on real-time AI deployment and practical business usability.
    """)
