
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

st.title("AI-Based Emotional Analysis of Customer Reviews")

review = st.text_area("Enter customer review")

if st.button("Analyze Emotion"):
    if review.strip() != "":
        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)
        emotion = label_map[prediction.item()]

        st.success(f"Detected Emotion: {emotion}")
    else:
        st.warning("Please enter a review")
