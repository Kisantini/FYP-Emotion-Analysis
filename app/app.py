import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
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

# -------------------------------------------------
# EMOTION KEYWORDS & REASON MAP
# -------------------------------------------------
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

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
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

# -------------------------------------------------
# DASHBOARD PAGE
# -------------------------------------------------
if menu == "🏠 Dashboard":
    st.title("CustomerSense AI")
    st.subheader("Understand What Your Customers Really Feel — In Real Time")

    st.markdown("""
    **CustomerSense AI** is a business-oriented emotion analysis system that helps
    organizations understand customer feedback beyond basic sentiment labels.
    """)

    st.markdown("### 🔍 Key Capabilities")
    st.markdown("""
    - Real-time customer review analysis  
    - Mixed-language and slang understanding  
    - Multi-emotion detection  
    - Sarcasm recognition  
    - Explainable business insights  
    """)

    st.info("Designed for business owners, customer service teams, and operations managers.")

# -------------------------------------------------
# LIVE REVIEW ANALYZER
# -------------------------------------------------
elif menu == "🧠 Live Review Analyzer":
    st.title("Live Review Analyzer")
    st.markdown("Analyze customer feedback instantly and identify emotional impact.")

    review = st.text_area(
        "Enter customer review",
        placeholder="Example: food ok but staff rude gila"
    )

    if st.button("Analyze Review"):
        if review.strip() != "":
            # ---------------- EXISTING ANALYSIS LOGIC ----------------
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1)

            primary_emotion = label_map[prediction.item()]
            detected_emotions = set()
            detected_emotions.add(primary_emotion)

            review_lower = review.lower()

            for emotion, keywords in emotion_keywords.items():
                for word in keywords:
                    if word in review_lower:
                        detected_emotions.add(emotion)

            # Sarcasm logic
            positive_triggers = ["very nice", "great", "excellent"]
            negative_triggers = ["broken", "damaged", "bad", "late", "lambat"]

            for p in positive_triggers:
                if p in review_lower:
                    for n in negative_triggers:
                        if n in review_lower:
                            detected_emotions.add("sarcasm")

            st.success(f"Detected Emotions: {', '.join(detected_emotions)}")

            reasons = set()
            for word, reason in reason_map.items():
                if word in review_lower:
                    reasons.add(reason)

            if reasons:
                st.info("Possible reasons:")
                for r in reasons:
                    st.write(f"- {r}")
            else:
                st.info("No specific issue detected")
            # ----------------------------------------------------------

        else:
            st.warning("Please enter a review")

# -------------------------------------------------
# INSIGHTS PAGE
# -------------------------------------------------
elif menu == "📊 Insights":
    st.title("Customer Insights Overview")
    st.markdown("""
    This section provides a high-level view of customer emotional trends.
    Advanced analytics and charts can be added in future versions.
    """)

    st.info("Analyze more customer reviews to generate insights.")

# -------------------------------------------------
# BUSINESS RECOMMENDATIONS PAGE
# -------------------------------------------------
elif menu == "💡 Business Recommendations":
    st.title("AI-Driven Business Recommendations")

    st.markdown("""
    CustomerSense AI supports management decision-making by translating
    customer emotions into actionable business improvements.
    """)

    st.markdown("""
    **Examples:**
    - Delivery delays → Improve logistics coordination  
    - Staff behaviour complaints → Customer service training  
    - Product damage issues → Enhance packaging and quality checks  
    """)

    st.success("This feature helps businesses respond proactively to customer dissatisfaction.")

# -------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------
elif menu == "ℹ️ About":
    st.title("About CustomerSense AI")

    st.markdown("""
    CustomerSense AI is developed using **BERT-based deep learning models**
    combined with rule-based reasoning to improve interpretability.

    The system is designed to:
    - Handle informal and mixed-language customer reviews  
    - Detect multiple emotions in real time  
    - Identify sarcasm and hidden dissatisfaction  
    - Provide explainable outputs for business users  

    This project is developed as a **Final Year Project (FYP)** focusing on
    practical, real-world AI deployment.
    """)
