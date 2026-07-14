import streamlit as st
import joblib

# Page Config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load Model
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Title
st.title("📰 Fake News Detection System")
st.markdown(
    "Paste a news article below and the model will predict whether it is **Real** or **Fake**."
)

# Text Area
news = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste your news article here..."
)

# Predict Button
if st.button("🔍 Analyze News"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        vector = tfidf.transform([news])

        prediction = model.predict(vector)[0]

        # Logistic Regression confidence
        confidence = model.predict_proba(vector).max() * 100

        st.divider()

        if prediction == 1:

            st.success("✅ REAL NEWS")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(float(confidence/100))

        else:

            st.error("🚨 FAKE NEWS")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(float(confidence/100))