
import streamlit as st
import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("model/bert_emotion_model")
    tokenizer = BertTokenizer.from_pretrained("model/bert_emotion_model")

    with open("model/bert_emotion_model/label_map.json") as f:
        label_map = json.load(f)

    reverse_label_map = {v: k for k, v in label_map.items()}
    return model, tokenizer, reverse_label_map

model, tokenizer, label_map = load_model()
model.eval()

emotion_keywords = {
    "anger": ["bad", "rude", "broken", "damage", "late", "lambat"],
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

st.title("AI-Based Emotional Analysis of Customer Reviews")

review = st.text_area("Enter customer review")

if st.button("Analyze Emotion"):
    if review.strip() != "":
        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)

        # Primary emotion from BERT
        primary_emotion = label_map[prediction.item()]
        detected_emotions = set()
        detected_emotions.add(primary_emotion)

        # Secondary emotions from keyword logic
        review_lower = review.lower()
        for emotion, keywords in emotion_keywords.items():
            for word in keywords:
                if word in review_lower:
                    detected_emotions.add(emotion)

        # Display detected emotions
        positive_triggers = ["very nice", "great", "excellent", "thanks a lot"]
        negative_triggers = ["broken", "damaged", "bad", "late", "lambat"]

        for p in positive_triggers:
            if p in review_lower:
                for n in negative_triggers:
                    if n in review_lower:
                        detected_emotions.add("sarcasm")
        st.success(f"Detected Emotions: {', '.join(detected_emotions)}")
        # ---------- STEP 11.2: Emotion Explanation ----------
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
        # --------------------------------------------------

    else:
        st.warning("Please enter a review")


