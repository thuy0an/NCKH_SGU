import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import re

# Tải bộ stopwords của NLTK để xử lý ngôn ngữ tự nhiên
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt') 
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

plt.rcParams['font.family'] = 'Arial'  
## đọc dataset và tóm tắt dữ liệu
df = pd.read_csv("Reviews.csv")


## Hiển thị thông tin tổng quan
# số dòng, số cột, kiểu dữ liệu của từng cột
print(f'+ Shape: {df.shape}')
print(df.info())
print(f'+ Contents')
display.display(df.head(5))


## Kiểm tra toàn vẹn dữ liệu (thiếu, trùng lặp)
print(f'Kiểm tra tính toàn vẹn dữ liệu')

print("\nDữ liệu trống:")
print(df.isnull().sum())
display.display(df[df.isnull().any(axis=1)])

print("\nSố đánh giá trùng lặp: ", df.duplicated(subset=['Text']).sum())
display.display(df[df.duplicated(subset=['Text'])])

## tiền xử lý dữ liệu
# Loại bỏ dòng bị thiếu & trùng lập
df = df.drop_duplicates(subset=['Text']).dropna()
print("\n✅ Đã loại bỏ các dòng trùng lặp và dữ liệu trống!")
print(f"\n✅ Số dòng sau tiền xử lý: {df.shape[0]}")

# Làm sạch văn bản nhanh hơn
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def fast_clean_text(text):
#     text = text.lower()  # Chuyển chữ thường
#     text = re.sub(r'[^\w\s]', '', text)  # Xóa dấu câu
#     words = word_tokenize(text)  # Tách từ
#     words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]  # Xóa stopwords + lemmatization
#     return " ".join(words)

# df['Text_Cleaned'] = df['Text'].map(fast_clean_text)
# # Lưu file kết quả
# df.to_csv("dataEDA.csv", index=False)
# print("\n✅ Dữ liệu đã lưu vào: dataEDA.csv 🎯")


df = pd.read_csv("dataEDA.csv")
# Phân tích đơn biến (Univariate Analysis)  
plt.figure(figsize=(8,5))
sns.countplot(x=df['Score'], palette="viridis")
plt.xlabel("Điểm đánh giá")
plt.ylabel("Số lượng")
plt.title("Phân phối điểm đánh giá")
plt.show()


## Phân tích phân phối cảm xúc  
def sentiment_label(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

df['Sentiment'] = df['Score'].apply(sentiment_label)
plt.figure(figsize=(6,6))
df['Sentiment'].value_counts().plot.pie(autopct="%1.1f%%", colors=['green', 'gray', 'red'])
plt.title("Phân phối cảm xúc")  
plt.ylabel("")  
plt.show()

## phân tích độ dài đánh giá Text
# Bước 1: Tạo cột mới lưu độ dài bài đánh giá (số từ trong mỗi đánh giá)
df['do_dai_danh_gia'] = df['Text_Cleaned'].apply(lambda x: len(x.split()))

# Bước 2: Vẽ biểu đồ phân phối độ dài đánh giá (Histogram)
plt.figure(figsize=(8,5))
sns.histplot(df['do_dai_danh_gia'], bins=50, kde=True, color='blue')
plt.xlabel("Số từ trong bài đánh giá")  # Trục X
plt.ylabel("Tần suất xuất hiện")  # Trục Y
plt.title("Phân phối độ dài bài đánh giá")  # Tiêu đề
plt.show()

# Bước 3: So sánh độ dài bài đánh giá theo cảm xúc (Boxplot)
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Sentiment'], y=df['do_dai_danh_gia'], palette="coolwarm")
plt.xlabel("Cảm xúc")  # Trục X
plt.ylabel("Độ dài bài đánh giá (số từ)")  # Trục Y
plt.title("Độ dài bài đánh giá theo nhóm cảm xúc")  # Tiêu đề
plt.show()

## phân tích tương quan
# biểu đồ phân tán scatter plot
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Score'], y=df['do_dai_danh_gia'], alpha=0.5)
plt.xticks([1, 2, 3, 4, 5]) 
plt.xlabel("Điểm đánh giá")
plt.ylabel("Độ dài bài đánh giá")
plt.title("Đánh giá chiều dài và điểm số")  # Tiêu đề biểu đồ
plt.show()
# ma trận tương quan correlation heatmap
correlation_matrix = df[['Score', 'do_dai_danh_gia']].corr()
plt.figure(figsize=(5,4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
