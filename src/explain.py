import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import joblib
import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("Loading model and data...")
model = joblib.load('models/model.pkl')
tfidf = joblib.load('models/tfidf.pkl')
X_test = joblib.load('models/X_test.pkl')

print("Calculating SHAP values (2-3 minutes)...")
explainer = shap.TreeExplainer(model)
X_sample = X_test[:100]
shap_values = explainer.shap_values(X_sample)

print("SHAP values shape:", np.array(shap_values).shape)

os.makedirs('reports', exist_ok=True)
feature_names = tfidf.get_feature_names_out()

sv = np.array(shap_values)
print("Detected shape:", sv.shape)

if len(sv.shape) == 3:
    sv_plot = sv[1]
elif len(sv.shape) == 2:
    sv_plot = sv
else:
    sv_plot = sv.reshape(100, -1)

print("Plotting with shape:", sv_plot.shape)

plt.figure(figsize=(12, 8))
shap.summary_plot(
    sv_plot,
    X_sample,
    feature_names=feature_names,
    max_display=20,
    show=False,
    plot_type="bar"
)
plt.title("Top 20 Words That Indicate a FAKE Job Posting")
plt.tight_layout()
plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved reports/shap_summary.png")

plt.figure(figsize=(12, 8))
shap.summary_plot(
    sv_plot,
    X_sample,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.title("SHAP Values - Word Impact on Fake Job Detection")
plt.tight_layout()
plt.savefig('reports/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved reports/shap_beeswarm.png")

print("\nAll SHAP charts saved to reports/ folder!")
