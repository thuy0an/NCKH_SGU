# User Sentiment Analysis on Product Reviews Using RoBERTa

## 1. Giới thiệu dự án  
Dự án của nhóm tập trung vào phân tích cảm xúc của người dùng dựa trên đánh giá sản phẩm từ tập dữ liệu **Amazon Fine Food Reviews**. Nhóm sử dụng mô hình **RoBERTa** để dự đoán cảm xúc (tích cực/ tiêu cực) từ khách hàng.

## 2. Lý do chọn đề tài  
Việc hiểu được đánh giá của khách hàng sẽ giúp doanh nghiệp cải thiện sản phẩm và dịch vụ. **RoBERTa**, là một mô hình mạnh dựa trên Transformer, với khả năng xử lý thông tin tốt hơn so với các phương pháp truyền thống.

## 3. Dữ liệu sử dụng  
**Nguồn:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  
**Thông tin:** Bao gồm hơn 500.000 đánh giá thực phẩm trên Amazon.

## 4. Phương pháp nghiên cứu  
- **Mô hình chính:** RoBERTa  
- **So sánh với các phương pháp khác:**  
  - Logistic Regression (nếu có thể)  
  - VADER (nếu có thể)  
- **Metrics đánh giá:** accuracy, precision, F1-Score

## 5. Công nghệ sử dụng  
- **Ngôn ngữ:** Python  
- **Thư viện chính:**  
 transformers, sklearn, NLTK, seaborn
