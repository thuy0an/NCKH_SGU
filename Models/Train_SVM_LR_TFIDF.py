import os
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# Trích xuất đặc trưng TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.5,
    sublinear_tf=True,
    norm='l2',
    smooth_idf=True
)
vectorizer.fit(train_preprocessed['Text_Cleaned'])
X_train = vectorizer.transform(train_preprocessed['Text_Cleaned'])
y_train = train_preprocessed['Score']
X_test = vectorizer.transform(test_data['Text_Cleaned'])
y_test = test_data['Score']

# Ensure the result directory exists
result_dir = os.path.join(os.getcwd(), 'Result')
os.makedirs(result_dir, exist_ok=True)

# 3. Huấn luyện và đánh giá mô hình SVM với LinearSVC
print("\nHuấn luyện mô hình SVM...")
# Sử dụng dual=False cho bộ dữ liệu có nhiều mẫu hơn đặc trưng
n_samples, n_features = X_train.shape
dual_param = n_samples <= n_features

svm_model = LinearSVC(
    C=0.1, 
    random_state=42, 
    dual=dual_param,
    max_iter=2000,  # Tăng số lần lặp
    verbose=1  # Hiển thị tiến trình
)
svm_model.fit(X_train, y_train)

# Đánh giá mô hình SVM
svm_predictions = svm_model.predict(X_test)
svm_report = classification_report(y_test, svm_predictions, output_dict=True)

svm_result_path = os.path.join(result_dir, 'svm_tfidf_results.txt')
with open(svm_result_path, 'w', encoding='utf-8') as f:
    f.write("Kết quả SVM (TF-IDF):\n")
    f.write(f"Accuracy: {accuracy_score(y_test, svm_predictions):.3f}\n")
    f.write(f"Precision: {svm_report['weighted avg']['precision']:.3f}\n")
    f.write(f"Recall: {svm_report['weighted avg']['recall']:.3f}\n")
    f.write(f"F1-score: {svm_report['weighted avg']['f1-score']:.3f}\n")

print("\nKết quả SVM (TF-IDF):")
print(f"Accuracy: {accuracy_score(y_test, svm_predictions):.3f}")
print(f"Precision: {svm_report['weighted avg']['precision']:.3f}")
print(f"Recall: {svm_report['weighted avg']['recall']:.3f}")
print(f"F1-score: {svm_report['weighted avg']['f1-score']:.3f}")

# 4. Huấn luyện và đánh giá mô hình Linear Regression
print("\nHuấn luyện mô hình Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_predictions_rounded = np.round(lr_predictions).astype(int)

lr_report = classification_report(y_test, lr_predictions_rounded, output_dict=True)

lr_result_path = os.path.join(result_dir, 'linear_regression_tfidf_results.txt')
with open(lr_result_path, 'w', encoding='utf-8') as f:
    f.write("Kết quả Linear Regression (TF-IDF):\n")
    f.write(f"Accuracy: {accuracy_score(y_test, lr_predictions_rounded):.3f}\n")
    f.write(f"Precision: {lr_report['weighted avg']['precision']:.3f}\n")
    f.write(f"Recall: {lr_report['weighted avg']['recall']:.3f}\n")
    f.write(f"F1-score: {lr_report['weighted avg']['f1-score']:.3f}\n")

print("\nKết quả Linear Regression (TF-IDF):")
print(f"Accuracy: {accuracy_score(y_test, lr_predictions_rounded):.3f}")
print(f"Precision: {lr_report['weighted avg']['precision']:.3f}")
print(f"Recall: {lr_report['weighted avg']['recall']:.3f}")
print(f"F1-score: {lr_report['weighted avg']['f1-score']:.3f}") 