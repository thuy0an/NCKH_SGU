# Tổng quan các bài báo

Bảng tổng hợp các bài báo đã khảo sát:

## 1. Thông Tin Các Bài Báo  

| **Bài báo**                                       | **Tác giả**                                         | **Index/Stock Country**                                     | **Model**                                | **Time Period** | **Performance Evaluation**                                                                                      | **Task**                  |
|---------------------------------------------------|----------------------------------------------------|------------------------------------------------------------|------------------------------------------|-----------------|----------------------------------------------------------------------------------------------------------------|----------------------------|
| **AFFR with BERT Model**                         | Xinyue Zhao <br> Yuandong Sun                     | Columbia University <br> Los Angeles                        | BERT                                     | 2022            | Accuracy = 79.82% <br> Loss = 0.5433                                    | Sentiment Classification |
| **Sentiment Classification for AFFR Using PySpark** | T.R. Aravidan <br> Vigneshwar C.N. <br> Suganeshwari Gopalswamy | IOS Press <br> Recent Developments in Electronics and Communication Systems | Logistic Regression <br> Naive Bayes  | 2023            | **Logistic Regression** <br> Accuracy = 87.61%  <br> Precision = 86.48%  <br> Recall = 88.21% <br> F1-Score = 87.23% <br> **Naive Bayes**  <br> Accuracy = 84.92%   <br> Precision = 81.27% <br> Recall = 83.92%  <br> F1-Score = 82.67% | Sentiment Classification |

---

## 2. Chi Tiết Mô Hình  

| **Bài báo**                                      | **Input**                         | **Output**                 | **Metric**                           | **Thành công**                                                  | **Hạn chế**                                    |
|-------------------------------------------------|----------------------------------|-----------------------------|--------------------------------------|----------------------------------------------------------------|------------------------------------------------|
| **AFFR with BERT Model**                        | Text review <br> Overall score  | Predicted Sentiment         | Accuracy <br> Loss                   | Sử dụng BERT để dự đoán điểm đánh giá từ nội dung văn bản, độ chính xác cao | Chưa kiểm tra trên các tập dữ liệu khác <br> Chưa so sánh với các mô hình khác |
| **Sentiment Classification for AFFR Using PySpark** | Text review, TF-IDF              | Predicted Sentiment         | Accuracy <br> Precision <br> Recall <br> F1-score | Sử dụng nhiều thuật toán ML khác nhau để so sánh <br> Xử lý big data bằng PySpark | Hiệu suất phụ thuộc vào Spark cluster |

---

