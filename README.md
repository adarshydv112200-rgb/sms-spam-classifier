# sms-spam-classifier

**Represented by:** Adarsh Yadav
**CSE (AIML) – 2nd Year**

Mini Project Report

## 1. Introduction
Spam messages are unsolicited and often misleading texts sent in bulk, commonly used for advertising, scams, or phishing attacks. This project builds a Machine Learning model that automatically classifies an SMS message as either **Spam** or **Ham** (a normal, non-spam message), using Natural Language Processing (NLP) techniques.

## 2. Objective
- To understand how text data can be converted into numerical features for Machine Learning.
- To train a classification model that can distinguish spam messages from genuine ones.
- To evaluate the model's performance using standard ML metrics.

## 3. Tools & Technology Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Libraries | pandas, scikit-learn |
| Feature Extraction | TF-IDF (Term Frequency - Inverse Document Frequency) |
| ML Algorithm | Multinomial Naive Bayes |

## 4. Methodology
1. **Dataset Preparation:** A labeled dataset of SMS messages (spam/ham) is used.
2. **Train-Test Split:** The data is split into training (75%) and testing (25%) sets.
3. **Feature Extraction:** Text messages are converted into numeric vectors using TF-IDF, which gives higher weight to important, distinguishing words.
4. **Model Training:** A Multinomial Naive Bayes classifier is trained on the TF-IDF features.
5. **Evaluation:** The trained model is tested and evaluated using Accuracy, Confusion Matrix, and a Classification Report (Precision, Recall, F1-score).
6. **Live Prediction:** The final model is used to classify new, unseen SMS messages in real time.

## 5. How Naive Bayes Works (Concept)
Naive Bayes is a probabilistic classifier based on Bayes' Theorem. It calculates the probability of a message being spam or ham based on the words it contains, assuming that each word contributes independently to the outcome. It is called 'naive' because it assumes word independence, which rarely holds true in real language — yet the algorithm still performs very well for text classification tasks like spam detection.

## 6. Source Code
The complete Python implementation is in [`spam_classifier-1.py`](./spam_classifier-1.py).

## 7. Sample Output

| Input Message | Prediction |
|---|---|
| Congratulations! You have won a free ticket, click now to claim | SPAM |
| Hi, can we meet tomorrow for the project discussion? | HAM |
| URGENT! Claim your cash prize before it expires today | SPAM |

The model achieved a reasonable accuracy on the sample dataset. Accuracy can be further improved by training on a larger, real-world dataset (e.g. the UCI SMS Spam Collection dataset with 5,500+ messages).

## 8. Conclusion
This mini project successfully demonstrates how Machine Learning and Natural Language Processing can be combined to automatically detect spam messages. The TF-IDF + Naive Bayes approach is simple, fast, and effective for text classification problems, making it a great starting point for understanding real-world AIML applications such as email filtering and content moderation.
