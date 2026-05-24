import streamlit as st
import joblib
import re
import ssl
import nltk

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="detective",
    layout="centered"
)

st.title("Fake Job Posting Detector")
st.markdown("Enter a job posting below and our AI will tell you if it looks **real or fake**.")

@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    tfidf = joblib.load('models/tfidf.pkl')
    return model, tfidf

model, tfidf = load_model()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stops = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stops]
    return ' '.join(words)

st.subheader("Enter Job Details")

title = st.text_input("Job Title", placeholder="e.g. Software Engineer")
company = st.text_area("Company Profile", placeholder="Tell us about the company...", height=100)
description = st.text_area("Job Description", placeholder="Paste the full job description here...", height=200)
requirements = st.text_area("Requirements", placeholder="List the job requirements...", height=100)

if st.button("Analyze Job Posting", type="primary"):
    if not description:
        st.warning("Please enter at least a job description.")
    else:
        combined = title + ' ' + company + ' ' + description + ' ' + requirements
        cleaned = clean_text(combined)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        st.divider()
        st.subheader("Result")

        if prediction == 1:
            st.error("FAKE JOB POSTING DETECTED!")
            st.metric("Confidence", f"{probability[1]*100:.1f}%")
            st.markdown("""
            **Red flags to watch out for:**
            - Vague job descriptions
            - Unrealistic salary promises
            - Requests for personal/financial information
            - No company details provided
            - Poor grammar and spelling
            """)
        else:
            st.success("This job posting looks REAL!")
            st.metric("Confidence", f"{probability[0]*100:.1f}%")
            st.markdown("""
            **Good signs in this posting:**
            - Clear job description
            - Realistic requirements
            - Professional language
            """)

st.divider()
st.markdown("Built with Random Forest + XGBoost | Trained on 17,880 job postings | 98.21% accuracy")
