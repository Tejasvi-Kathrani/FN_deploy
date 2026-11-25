import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load old and new datasets
old_data = pd.read_csv("FakeNewsNet.csv")
new_data = pd.read_csv("new_labeled_news.csv")

# Select required columns
old_data = old_data[['title', 'real']]
new_data = new_data[['title', 'real']]

# Combine datasets
df = pd.concat([old_data, new_data], ignore_index=True)

# Clean data
df = df.dropna(subset=['title', 'real'])
df['title'] = df['title'].astype(str)
df['real'] = df['real'].astype(int)

# Features and target
X = df['title']
y = df['real']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=2000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Load existing model architecture
old_model = joblib.load("verifact_model.pkl")

# Create fresh model
new_model = clone(old_model)

# Train model
new_model.fit(X_train_tfidf, y_train)

# Evaluate
predictions = new_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("New Model Accuracy:", accuracy)

# Save updated model
joblib.dump(new_model, "verifact_model.pkl")
joblib.dump(tfidf, "verifact_tfidf.pkl")

print("VeriFact Model Retrained Successfully!")