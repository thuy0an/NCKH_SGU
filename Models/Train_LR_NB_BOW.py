import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import os
print("Current working dir:", os.getcwd())


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 1. Đọc dữ liệu đã xử lý
train_preprocessed = pd.read_csv('NCKH_SGU/TrichXuatDacTrung/TrainPreProcess.csv')
test_data = pd.read_csv('NCKH_SGU/TrainAndTestData/test.csv')



def preprocess_Text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Áp dụng tiền xử lý cho test
test_data['Text_Cleaned'] = test_data['Text'].apply(preprocess_Text)

# Trích xuất đặc trưng BoW cho test
vectorizer = CountVectorizer(max_features=5000)
vectorizer.fit(train_preprocessed['Text_Cleaned'])
X_train = vectorizer.transform(train_preprocessed['Text_Cleaned'])
y_train = train_preprocessed['Score'] 
X_test = vectorizer.transform(test_data['Text_Cleaned'])
y_test = test_data['Score']

# 3. Huấn luyện và đánh giá mô hình Logistic Regression
print("Huấn luyện mô hình Logistic Regression...")
lr_model = LogisticRegression(solver='saga' ,max_iter=5000, random_state=42)
lr_model.fit(X_train, y_train)

# Đánh giá mô hình Logistic Regression
lr_predictions = lr_model.predict(X_test)
lr_report = classification_report(y_test, lr_predictions, output_dict=True)

# Ensure the result directory exists
result_dir = os.path.join(os.getcwd(), 'Result')
os.makedirs(result_dir, exist_ok=True)

# Save Logistic Regression results
lr_result_path = os.path.join(result_dir, 'logistic_regression_results.txt')
with open(lr_result_path, 'w', encoding='utf-8') as f:
    f.write("Kết quả Logistic Regression:\n")
    f.write(f"Accuracy: {accuracy_score(y_test, lr_predictions):.3f}\n")
    f.write(f"Precision: {lr_report['weighted avg']['precision']:.3f}\n")
    f.write(f"Recall: {lr_report['weighted avg']['recall']:.3f}\n")
    f.write(f"F1-score: {lr_report['weighted avg']['f1-score']:.3f}\n")

print("\nKết quả Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, lr_predictions):.3f}")
print(f"Precision: {lr_report['weighted avg']['precision']:.3f}")
print(f"Recall: {lr_report['weighted avg']['recall']:.3f}")
print(f"F1-score: {lr_report['weighted avg']['f1-score']:.3f}")

# 4. Huấn luyện và đánh giá mô hình Naive Bayes
print("\nHuấn luyện mô hình Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Đánh giá mô hình Naive Bayes
nb_predictions = nb_model.predict(X_test)
nb_report = classification_report(y_test, nb_predictions, output_dict=True)

# Save Naive Bayes results
nb_result_path = os.path.join(result_dir, 'naive_bayes_results.txt')
with open(nb_result_path, 'w', encoding='utf-8') as f:
    f.write("Kết quả Naive Bayes:\n")
    f.write(f"Accuracy: {accuracy_score(y_test, nb_predictions):.3f}\n")
    f.write(f"Precision: {nb_report['weighted avg']['precision']:.3f}\n")
    f.write(f"Recall: {nb_report['weighted avg']['recall']:.3f}\n")
    f.write(f"F1-score: {nb_report['weighted avg']['f1-score']:.3f}\n")

print("\nKết quả Naive Bayes:")
print(f"Accuracy: {accuracy_score(y_test, nb_predictions):.3f}")
print(f"Precision: {nb_report['weighted avg']['precision']:.3f}")
print(f"Recall: {nb_report['weighted avg']['recall']:.3f}")
print(f"F1-score: {nb_report['weighted avg']['f1-score']:.3f}")