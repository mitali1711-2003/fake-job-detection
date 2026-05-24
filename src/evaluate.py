import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)
import os

print("Loading model...")
model = joblib.load('models/model.pkl')
X_test = joblib.load('models/X_test.pkl')
y_test = joblib.load('models/y_test.pkl')

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n===== MODEL RESULTS =====")
print(classification_report(y_test, y_pred, target_names=['Real Job', 'Fake Job']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

os.makedirs('reports', exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Fake Job Detection')
plt.savefig('reports/confusion_matrix.png')
print("\nConfusion matrix saved to reports/confusion_matrix.png")
