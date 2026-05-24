import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import joblib
import re
import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("Fake Job Posting Detector")
st.markdown("Enter a job posting below and our AI will tell you if it looks **real or fake** — and explain **why**.")

@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    tfidf = joblib.load('models/tfidf.pkl')
    explainer = shap.TreeExplainer(model)
    return model, tfidf, explainer

model, tfidf, explainer = load_model()

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
        st.subheader("Why did the AI decide this?")
        st.markdown("These are the **top words** that influenced the decision:")

        with st.spinner("Calculating word importance..."):
            shap_values = explainer.shap_values(vectorized)
            sv = np.array(shap_values)

            if len(sv.shape) == 3:
                sv_single = sv[1][0]
            else:
                sv_single = sv[0]

            feature_names = tfidf.get_feature_names_out()
            indices = np.argsort(np.abs(sv_single))[-15:][::-1]

            top_words = [(feature_names[i], sv_single[i]) for i in indices if sv_single[i] != 0]

            if top_words:
                fig, ax = plt.subplots(figsize=(8, 5))
                words = [w[0] for w in top_words]
                values = [w[1] for w in top_words]
                colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]
                bars = ax.barh(words[::-1], values[::-1], color=colors[::-1])
                ax.axvline(x=0, color='black', linewidth=0.5)
                ax.set_xlabel("SHAP Value (red = fake indicator, green = real indicator)")
                ax.set_title("Word Importance for This Prediction")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown("**Red bars** = words pushing towards FAKE")
                st.markdown("**Green bars** = words pushing towards REAL")
            else:
                st.info("Not enough distinctive words found for explanation.")

st.divider()
st.markdown("Built with XGBoost + SHAP | Trained on 17,880 job postings | 98.21% accuracy")
