# Fake Job Detection using Machine Learning

A machine learning project that detects fraudulent job postings using NLP and multiple ML classifiers.

## Live Demo
Try the app here: https://mitali1711-2003-fake-job-detection-app-kz7td1.streamlit.app

## Model Comparison
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 97.48% |
| Random Forest | 97.99% |
| **XGBoost** | **98.21%** |

XGBoost performed best and was selected as the final model.

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

## EDA Charts
- Class distribution (real vs fake)
- Top 10 industries with fake jobs
- Word cloud for fake job descriptions
- Word cloud for real job descriptions
- Model accuracy comparison

## Project Structure
fake-job-detection/
├── data/               <- dataset files
├── notebooks/
│   └── eda.py          <- generates all EDA charts
├── src/
│   ├── preprocess.py   <- cleans text data
│   ├── train.py        <- trains and compares 3 models
│   └── evaluate.py     <- evaluates best model
├── models/             <- saved model files
├── reports/            <- all 5 charts saved here
├── app.py              <- Streamlit web app
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
streamlit run app.py
```

## Tech Stack
- Python, Pandas, Scikit-learn
- TF-IDF Vectorization
- Random Forest, Logistic Regression, XGBoost
- SMOTE for class imbalance
- NLTK for text cleaning
- Matplotlib, Seaborn, WordCloud
- Streamlit for web app
