# Fake Job Detection using Machine Learning

A machine learning project that detects fraudulent job postings using NLP and Random Forest classifier.

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 98% |
| ROC-AUC | 0.9781 |
| Fake Job Precision | 0.99 |
| Fake Job Recall | 0.61 |
| F1-Score | 0.75 |

## Dataset
- Source: [Kaggle - Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- 17,880 job postings (17,014 real + 866 fake)

## Project Structure
fake-job-detection/
├── data/          <- dataset files
├── notebooks/     <- EDA notebooks
├── src/
│   ├── preprocess.py   <- cleans text data
│   ├── train.py        <- trains the model
│   └── evaluate.py     <- evaluates results
├── models/        <- saved model files
├── reports/       <- confusion matrix chart
└── requirements.txt
## How to Run
```bash
git clone https://github.com/mitali1711-2003/fake-job-detection.git
cd fake-job-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/preprocess.py
python3 src/train.py
python3 src/evaluate.py
```

## Tech Stack
- Python, Pandas, Scikit-learn
- TF-IDF Vectorization
- Random Forest Classifier
- SMOTE for class imbalance
- NLTK for text cleaning
