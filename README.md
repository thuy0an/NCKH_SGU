# User Sentiment Analysis on Product Reviews Using Machine Learning

## 1. Giới thiệu dự án  
Dự án của nhóm tập trung vào đánh giá hiệu suất của các thuật toán Machine Learning trong phân tích cảm xúc của người dùng dựa trên đánh giá sản phẩm từ tập dữ liệu **Amazon Fine Food Reviews (AFFR)**. Các mô hình có thể sử dụng bao gồm Logistic Regression, SVM, Linear Regression, NaiveBayes, VADER, với mục tiêu dự đoán cảm xúc (tích cực/tiêu cực) từ phản hồi khách hàng.

## 2. Lý do chọn đề tài  
Việc hiểu được đánh giá của khách hàng giúp doanh nghiệp cải thiện sản phẩm và dịch vụ. Nhóm thực hiện nghiên cứu để **so sánh hiệu suất của các thuật toán Machine Learning** trên tập dữ liệu lớn, từ đó xác định phương pháp phù hợp nhất trong phân tích cảm xúc.

## 3. Dữ liệu sử dụng  
**Nguồn:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) 
**Thông tin:** Bao gồm hơn 500.000 đánh giá thực phẩm trên Amazon, gồm văn bản đánh giá, số sao và trạng thái tích cực/ tiêu cực/ trung lập.

## 4. Phương pháp nghiên cứu  
### Mô hình Machine Learning thử nghiệm  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Linear Regression
- Naive Bayes

### Tiền xử lý dữ liệu
- Loại bỏ các dữ liệu thừa, các cột không cần thiết trong việc phân tích, các dữ liệu thiếu
- Loại bỏ dữ liệu trung lặp (điểm 3)
- Sử dụng BoW hoặc TF-IDF trích xuất đặc trưng (nếu có thể)
### Metrics đánh giá mô hình
- Accuracy
- Percision
- Recall
- F1-Score

### Công nghệ sử dụng
- **Ngôn ngữ:** Python  
- **Thư viện chính:**  
- `sklearn`, `NLTK`, `seaborn`, `matplotlib`, `pandas`, `numpy`
