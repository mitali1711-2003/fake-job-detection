# Fake Job Detection using Machine Learning

An end-to-end ML project that detects fraudulent job postings using NLP, XGBoost, and SHAP explainability.

## Live Demo
Try the app: https://mitali1711-2003-fake-job-detection-app-kz7td1.streamlit.app

## What makes this advanced
- Compares 3 ML models and picks the best one automatically
- SHAP explainability shows exactly WHICH WORDS made the model decide fake or real
- Fully deployed live web app anyone can use
- 5 EDA charts showing data insights

## Model Comparison
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 97.48% |
| Random Forest | 97.99% |
| **XGBoost** | **98.21%** |

## Final Results (XGBoost)
| Metric | Score |
|--------|-------|
| Accuracy | 98.21% |
| ROC-AUC | 0.9781 |
| Fake Job Precision | 0.99 |
| Fake Job Recall | 0.61 |
| F1-Score | 0.75 |

## Dataset
- Source: [Kaggle - Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- 17,880 job postings (17,014 real + 866 fake)

## Project Structure
fake-job-detection/
├── data/               <- dataset files
├── notebooks/
│   └── eda.py          <- 5 EDA charts
├── src/
│   ├── preprocess.py   <- cleans text data
│   ├── train.py        <- trains and compares 3 models
│   ├── evaluate.py     <- evaluates best model
│   └── explain.py      <- SHAP explainability charts
├── models/             <- saved model files
├── reports/            <- all charts including SHAP
├── app.py              <- Streamlit web app with SHAP
└── requirements.txt
## How to Run Locally
```bash
git clone https://github.com/mitali1711-2003/fake-job-detection.git
cd fake-job-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/preprocess.py
python3 src/train.py
python3 src/evaluate.py
python3 src/explain.py
streamlit run app.py
```

## Tech Stack
- Python, Pandas, Scikit-learn
- TF-IDF Vectorization
- Random Forest, Logistic Regression, XGBoost
- SMOTE for class imbalance
- SHAP for model explainability
- NLTK for text cleaning
- Matplotlib, Seaborn, WordCloud
- Streamlit for web app deployment
