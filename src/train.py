import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

df = pd.read_csv('data/cleaned.csv')

X = df['text']
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Converting text to numbers using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Balancing classes using SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

best_score = 0
best_model_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_balanced, y_train_balanced)
    score = model.score(X_test_tfidf, y_test)
    print(f"{name} accuracy: {score:.4f}")
    if score > best_score:
        best_score = score
        best_model_name = name
        best_model = model

print(f"\nBest model: {best_model_name} with accuracy {best_score:.4f}")

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/model.pkl')
joblib.dump(tfidf, 'models/tfidf.pkl')
joblib.dump(X_test_tfidf, 'models/X_test.pkl')
joblib.dump(y_test, 'models/y_test.pkl')

print("Best model saved!")
