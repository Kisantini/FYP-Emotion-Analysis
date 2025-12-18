import streamlit as st
import pandas as pd
import re
import random
import time

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="CustomerSense AI",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# SESSION STATE
# ======================================================
if "role" not in st.session_state:
    st.session_state.role = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "users" not in st.session_state:
    st.session_state.users = {
        "admin@example.com": "Admin@123"
    }

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def valid_password(password):
    return (
        len(password) >= 6 and
        re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)
    )

def fake_emotion_analysis(text):
    text = text.lower()
    emotions = []

    if any(x in text for x in ["lambat", "late", "slow"]):
        emotions.append("Anger")
    if any(x in text for x in ["rude", "kasar", "bad"]):
        emotions.append("Disappointment")
    if any(x in text for x in ["nice", "good", "friendly"]):
        emotions.append("Happiness")
    if any(x in text for x in ["broken", "damaged"]):
        emotions.append("Anger")
        emotions.append("Sarcasm")

    if not emotions:
        emotions.append("Neutral")

    return list(set(emotions))

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<h1 style='text-align:center;'>CustomerSense AI</h1>
<p style='text-align:center; font-size:18px;'>
AI-Based Emotional Analysis of Customer Reviews
</p>
<hr>
""", unsafe_allow_html=True)

# ======================================================
# LANDING PAGE
# ======================================================
if st.session_state.role is None:
    st.markdown("## Choose Your Role")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Customer")
        st.write("Quickly share your feedback")
        if st.button("Continue as Customer"):
            st.session_state.role = "customer"

    with col2:
        st.markdown("### 🏢 Business User")
        st.write("Analyze customer emotions & insights")
        if st.button("Continue as Business User"):
            st.session_state.role = "business"

# ======================================================
# CUSTOMER FLOW
# ======================================================
if st.session_state.role == "customer":
    st.markdown("## 📝 Customer Feedback")

    feedback = st.text_area(
        "Please share your experience",
        height=150
    )

    if st.button("Submit Feedback"):
        if feedback.strip():
            st.success(
                "THANK YOU FOR YOUR TIME.\n\n"
                "SEE YOU AGAIN.\n\n"
                "HAVE A NICE DAY."
            )
            time.sleep(2)
            st.session_state.role = None
        else:
            st.warning("Please write your feedback.")

# ======================================================
# BUSINESS LOGIN
# ======================================================
if st.session_state.role == "business" and not st.session_state.logged_in:
    st.markdown("## 🔐 Business Login")

    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            if email in st.session_state.users and st.session_state.users[email] == password:
                st.session_state.logged_in = True
                st.success("Login successful")
            else:
                st.error("Incorrect email or password")

    with col2:
        if st.button("Forgot Password"):
            otp = random.randint(100000, 999999)
            st.info(f"OTP sent to email (Simulation): {otp}")

    st.markdown("---")
    st.markdown("### 🆕 Create New Account")

    new_email = st.text_input("New Email")
    new_password = st.text_input("New Password", type="password")

    if st.button("Register"):
        if not valid_password(new_password):
            st.error("Password must be at least 6 characters and contain a special character")
        else:
            st.session_state.users[new_email] = new_password
            st.success("Account created successfully")

# ======================================================
# BUSINESS DASHBOARD
# ======================================================
if st.session_state.logged_in:
    st.sidebar.title("📊 Menu")
    menu = st.sidebar.radio(
        "Navigate",
        ["Single Review", "Batch Reviews", "Upload File", "Dashboard", "Logout"]
    )

    # ---------------- SINGLE REVIEW ----------------
    if menu == "Single Review":
        st.markdown("## 🧠 Single Review Analysis")

        review = st.text_area("Enter customer review")

        if st.button("Analyze Review"):
            if review.strip():
                emotions = fake_emotion_analysis(review)
                st.success(f"Detected Emotions: {', '.join(emotions)}")
            else:
                st.warning("Please enter a review")

    # ---------------- BATCH REVIEWS ----------------
    if menu == "Batch Reviews":
        st.markdown("## 📑 Batch Review Analysis")

        reviews = st.text_area(
            "Enter multiple reviews (one per line)",
            height=200
        )

        if st.button("Analyze Batch"):
            review_list = reviews.split("\n")
            for r in review_list:
                if r.strip():
                    emotions = fake_emotion_analysis(r)
                    st.write(f"🔹 {r}")
                    st.write(f"Emotion: {', '.join(emotions)}")
                    st.markdown("---")

    # ---------------- FILE UPLOAD ----------------
    if menu == "Upload File":
        st.markdown("## 📂 Upload Reviews File")

        uploaded_file = st.file_uploader(
            "Upload CSV or TXT file",
            type=["csv", "txt"]
        )

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                reviews = df.iloc[:, 0].astype(str).tolist()
            else:
                reviews = uploaded_file.read().decode("utf-8").splitlines()

            st.success(f"{len(reviews)} reviews loaded")

            if st.button("Analyze File"):
                for r in reviews:
                    emotions = fake_emotion_analysis(r)
                    st.write(f"🔹 {r}")
                    st.write(f"Emotion: {', '.join(emotions)}")
                    st.markdown("---")

    # ---------------- DASHBOARD ----------------
    if menu == "Dashboard":
        st.markdown("## 📈 Emotion Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", "120")
        col2.metric("Negative Emotions", "45")
        col3.metric("Positive Emotions", "75")

        st.info("This dashboard summarizes customer emotional trends.")

    # ---------------- LOGOUT ----------------
    if menu == "Logout":
        st.session_state.logged_in = False
        st.session_state.role = None
        st.success("Logged out successfully")
