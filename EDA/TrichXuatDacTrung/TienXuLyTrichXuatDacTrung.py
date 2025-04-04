import pandas as pd
import re
from IPython import display
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Tải các gói cần thiết NLTK
nltk.download('stopwords')
nltk.download('punkt') 
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Đọc dữ liệu dataset
df = pd.read_csv("Reviews.csv")

""" Hàm tiền xử lý dữ liệu """
def TienXuLyDuLieu(text):
    text = text.lower()  # Chuyển chữ thường
    text = re.sub(r'<br\s*/?>', ' ', text)  # Xóa thẻ <br> hoặc <br/> và thay thế bằng khoảng trắng
    text = re.sub(r'[^\w\s]', '', text)  # Xóa dấu câu
    words = word_tokenize(text)  # Tách từ
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]  # Xóa stopwords + lemmatization
    return " ".join(words)


""" Lấy các mẫu nhỏ trong dataset để tìm đặc trưng"""
one_1500 = df[df['Score'] == 1].sample(n=1500)
two_500 = df[df['Score'] == 2].sample(n=500)
three_500 = df[df['Score'] == 3].sample(n=500)
four_500 = df[df['Score'] == 4].sample(n=500)
five_1500 = df[df['Score'] == 5].sample(n=1500)

CacMauThuNghiem = pd.concat([one_1500, two_500, three_500, four_500, five_1500], axis=0)
plt.figure(figsize=(8, 5))
sns.countplot(x='Score', data=CacMauThuNghiem)
plt.title('Phân phối điểm đánh giá trong tập dữ liệu mẫu')
plt.show()


""" Thông tin dữ liệu sau khi tiền xử lý"""
# Tách một ví dụ từ cột 'Text'
example = CacMauThuNghiem['Text'].iloc[49]
print("Cột Text phiên bản gốc:\n", example)

# Tokenize và in kết quả
token = nltk.word_tokenize(example)
print("Tokenized words:\n", token)

# Áp dụng tiền xử lý văn bản cho cột 'Text' và in ra kết quả
CacMauThuNghiem['Text_Cleaned'] = CacMauThuNghiem['Text'].map(TienXuLyDuLieu)
print(f"\nVăn bản sau khi làm sạch:\n{CacMauThuNghiem['Text_Cleaned'].iloc[49]}")


""" Trích xuất đặc trưng 5 mẫu nhỏ"""
# Lấy mẫu 5 dòng dữ liệu từ Dataset
sampled_reviews = CacMauThuNghiem.sample(n=5, random_state=42)
sampled_reviews['Text_Length'] = sampled_reviews['Text'].apply(len)
sampled_reviews['Word_Count'] = sampled_reviews['Text_Cleaned'].apply(lambda x: len(x.split()))

## Trích xuất đặc trưng BoW
vectorizer = CountVectorizer(max_features=20)  # Giới hạn số lượng đặc trưng để dễ nhìn thấy kết quả
bow_matrix = vectorizer.fit_transform(sampled_reviews['Text_Cleaned'])

# Hiển thị thông tin của 5 dòng đã trích xuất đặc trưng BoW
print("\nThông tin của 5 dòng đã trích xuất đặc trưng BoW:")
print(sampled_reviews['Text_Cleaned'])

## Hiển thị các từ trong từ điển BoW
feature_names = vectorizer.get_feature_names_out()
print("\nCác từ trong từ điển BoW:")
print(feature_names)

## In ra ma trận BoW trích xuất đặc trưng của 5 dòng
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)
bow_df.index = sampled_reviews.index
df_features = pd.concat([sampled_reviews[['Text_Length', 'Word_Count']], bow_df], axis=1)

print(f"\nĐặc trưng đã trích xuất: {df_features.shape[1]} đặc trưng")
print(df_features.head())


""" Trích xuất đặc trưng cho toàn bộ các mẫu thử nghiệm"""
# Trích xuất đặc trưng BoW cho toàn bộ tập dữ liệu với nhiều đặc trưng hơn
full_vectorizer = CountVectorizer(max_features=1000)
full_bow_matrix = full_vectorizer.fit_transform(CacMauThuNghiem['Text_Cleaned'])

print(f"Kích thước ma trận BoW: {full_bow_matrix.shape}")

# Top từ xuất hiện nhiều nhất
feature_names = full_vectorizer.get_feature_names_out()
count_sum = full_bow_matrix.sum(axis=0)
word_freq = [(word, count_sum[0, idx]) for idx, word in enumerate(feature_names)]
word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

print("\nTop 20 từ xuất hiện nhiều nhất:")
for word, freq in word_freq[:20]:
    print(f"{word}: {freq}")

top_words = pd.DataFrame(word_freq[:20], columns=['word', 'frequency'])
plt.figure(figsize=(12, 8))
sns.barplot(x='frequency', y='word', data=top_words)
plt.title('Top 20 từ xuất hiện nhiều nhất trong đánh giá')
plt.tight_layout()
plt.show()


