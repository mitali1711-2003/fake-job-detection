import pandas as pd
import re
import ssl
import nltk

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

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

df = pd.read_csv('data/fake_job_postings.csv')

df['text'] = (
    df['title'].fillna('') + ' ' +
    df['company_profile'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['requirements'].fillna('')
)

df['text'] = df['text'].apply(clean_text)
df[['text', 'fraudulent']].to_csv('data/cleaned.csv', index=False)

print("Done! Cleaned data saved to data/cleaned.csv")
print(df['fraudulent'].value_counts())
