"""
Mini Project: SMS Spam Classifier using Machine Learning
CSEP Lab - Represented by Adarsh Yadav, CSE (AIML) - 2nd Year

Approach:
1. Load labeled SMS dataset (message, label)
2. Split data into train/test sets
3. Convert text to numeric features using TF-IDF
4. Train a Multinomial Naive Bayes classifier
5. Evaluate using accuracy, confusion matrix, classification report
6. Predict new/unseen SMS messages
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# 1. Load dataset (message, label)
# ---------------------------------------------------------
df = pd.read_csv("spam.csv")
print("Dataset loaded successfully. Total messages:", len(df))
print(df["label"].value_counts())
print()

# ---------------------------------------------------------
# 2. Split into train/test sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.25, random_state=42
)

# ---------------------------------------------------------
# 3. Convert text to TF-IDF features
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------------
# 4. Train Naive Bayes model
# ---------------------------------------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------------
# 5. Evaluate the model
# ---------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 6. Predict new / unseen messages (Live Prediction)
# ---------------------------------------------------------
sample_messages = [
    "Congratulations! You won a free prize, click now",
    "Hi, can we meet tomorrow for the project discussion?",
    "URGENT! Claim your cash prize before it expires today"
]

predictions = model.predict(vectorizer.transform(sample_messages))

print("\nSample Predictions:")
for msg, label in zip(sample_messages, predictions):
    print(f"'{msg}' -> {label.upper()}")
