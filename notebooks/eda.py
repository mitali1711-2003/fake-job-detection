import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

os.makedirs('reports', exist_ok=True)

df = pd.read_csv('data/fake_job_postings.csv')

# Chart 1 - Class imbalance
plt.figure(figsize=(6,4))
counts = df['fraudulent'].value_counts()
plt.bar(['Real Jobs', 'Fake Jobs'], counts.values, color=['#2ecc71','#e74c3c'])
plt.title('Real vs Fake Job Postings')
plt.ylabel('Count')
for i, v in enumerate(counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/class_distribution.png')
plt.close()
print("Saved class distribution chart")

# Chart 2 - Top industries with fake jobs
plt.figure(figsize=(10,5))
fake_industries = df[df['fraudulent']==1]['industry'].value_counts().head(10)
sns.barplot(x=fake_industries.values, y=fake_industries.index, palette='Reds_r')
plt.title('Top 10 Industries with Fake Jobs')
plt.xlabel('Number of Fake Postings')
plt.tight_layout()
plt.savefig('reports/fake_industries.png')
plt.close()
print("Saved fake industries chart")

# Chart 3 - Word cloud for fake jobs
fake_text = ' '.join(df[df['fraudulent']==1]['description'].dropna().tolist())
wordcloud = WordCloud(width=800, height=400,
                      background_color='white',
                      colormap='Reds').generate(fake_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in Fake Job Descriptions')
plt.tight_layout()
plt.savefig('reports/fake_wordcloud.png')
plt.close()
print("Saved fake jobs word cloud")

# Chart 4 - Word cloud for real jobs
real_text = ' '.join(df[df['fraudulent']==0]['description'].dropna().tolist())
wordcloud2 = WordCloud(width=800, height=400,
                       background_color='white',
                       colormap='Greens').generate(real_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in Real Job Descriptions')
plt.tight_layout()
plt.savefig('reports/real_wordcloud.png')
plt.close()
print("Saved real jobs word cloud")

# Chart 5 - Model comparison
plt.figure(figsize=(7,4))
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
scores = [0.9748, 0.9799, 0.9821]
colors = ['#3498db','#2ecc71','#e74c3c']
bars = plt.bar(models, scores, color=colors)
plt.ylim(0.96, 0.99)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.0003,
             f'{score:.4f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/model_comparison.png')
plt.close()
print("Saved model comparison chart")

print("\nAll 5 charts saved to reports/ folder!")
