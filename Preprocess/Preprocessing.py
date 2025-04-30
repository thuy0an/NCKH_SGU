import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.model_selection import train_test_split
from IPython import display

# Tải các gói cần thiết NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Đường dẫn đến file dữ liệu
data_path = "Reviews.csv"

# Đọc dữ liệu từ file CSV
df = pd.read_csv(data_path)
print("Dữ liệu gốc:")
# số dòng, số cột, kiểu dữ liệu của từng cột
print(f'+ Shape: {df.shape}')
print(df.info())
print(f'+ Contents')
display.display(df.head(5))

# Hiển thị phân phối điểm đánh giá
plt.figure(figsize=(8, 5))
sns.countplot(x='Score', data=df)
plt.title('Phân phối điểm đánh giá trong tập dữ liệu')
plt.show()

# Chia dữ liệu thành tập train và test với tỉ lệ 70-30
train, test = train_test_split(df, test_size=0.3, random_state=42)

# Tạo thư mục lưu trữ nếu chưa tồn tại
output_dir = "TrainAndTestData"
os.makedirs(output_dir, exist_ok=True)

# Lưu dữ liệu train và test vào các file riêng biệt
train_file = os.path.join(output_dir, "train.csv")
test_file = os.path.join(output_dir, "test.csv")

train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)

print(f"Dữ liệu train đã được lưu tại: {train_file}")
print(f"Dữ liệu test đã được lưu tại: {test_file}")

# Kiểm tra lại dữ liệu sau khi lưu
print("\nKiểm tra dữ liệu train sau khi lưu:")
train_check = pd.read_csv(train_file)
print(train_check.head())

print("\nKiểm tra dữ liệu test sau khi lưu:")
test_check = pd.read_csv(test_file)
print(test_check.head())

# Tiền xử lý dữ liệu từ train.csv
print("\nBắt đầu tiền xử lý dữ liệu từ train.csv...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

train_check['Text_Cleaned'] = train_check['Text'].map(preprocess_text)

# Lưu kết quả tiền xử lý vào TrainPreProcess.csv
preprocess_dir = "TrichXuatDacTrung"
os.makedirs(preprocess_dir, exist_ok=True)
preprocess_file = os.path.join(preprocess_dir, "TrainPreProcess.csv")
train_check.to_csv(preprocess_file, index=False)
print(f"Dữ liệu sau tiền xử lý đã được lưu tại: {preprocess_file}")

# Hiển thị ví dụ trước và sau khi tiền xử lý
example = train_check['Text'].iloc[0]
print("\nVí dụ trước khi tiền xử lý:")
print(example)
print("\nVí dụ sau khi tiền xử lý:")
print(preprocess_text(example))

# Trích xuất đặc trưng BoW
print("\nBắt đầu trích xuất đặc trưng BoW...")
vectorizer = CountVectorizer(max_features=5000)
bow_matrix = vectorizer.fit_transform(train_check['Text_Cleaned'])

# Hiển thị thông tin của 5 dòng đã trích xuất đặc trưng BoW
print("\nThông tin của 5 dòng đã trích xuất đặc trưng BoW:")
print(train_check['Text_Cleaned'].head(5))

# Hiển thị các từ trong từ điển BoW
feature_names = vectorizer.get_feature_names_out()
print("\nCác từ trong từ điển BoW:")
print(feature_names[:20])  # Hiển thị 20 từ đầu tiên

# In ra ma trận BoW trích xuất đặc trưng của 5 dòng
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)
bow_df.index = train_check.index
train_check['Text_Length'] = train_check['Text'].apply(len)
train_check['Word_Count'] = train_check['Text_Cleaned'].apply(lambda x: len(x.split()))
df_features = pd.concat([train_check[['Text_Length', 'Word_Count']], bow_df], axis=1)

print(f"\nĐặc trưng đã trích xuất: {df_features.shape[1]} đặc trưng")
print(df_features.head())

# Lưu kết quả đặc trưng vào FeatureBoW.csv
feature_file = os.path.join(preprocess_dir, "FeatureBoW.csv")
df_features.to_csv(feature_file, index=False)
print(f"Đặc trưng BoW đã được lưu tại: {feature_file}")

# Hiển thị biểu đồ tần suất từ xuất hiện nhiều nhất
word_freq = bow_matrix.sum(axis=0).A1
word_freq_df = pd.DataFrame({'word': feature_names, 'frequency': word_freq})
top_words = word_freq_df.nlargest(20, 'frequency')

plt.figure(figsize=(12, 8))
sns.barplot(x='frequency', y='word', data=top_words)
plt.title('Top 20 từ xuất hiện nhiều nhất trong đánh giá')
plt.tight_layout()
plt.show()


