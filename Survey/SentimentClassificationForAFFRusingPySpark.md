# Sentiment Classification for AFFR using PySpark

## Thông tin bài báo
- **Tác giả**: T.R. Aravidan, Vigneshwar C.N., Suganeshwari Gopalswamy  
- **Năm**: 2023  
- **Nguồn**: Recent Developments in Electronics and Communication Systems  
- **Mô hình**: Các thuật toán Machine Learning với PySpark  
- **Dataset**: Amazon Fine Food Reviews  
- **Link**: [Sentiment Classification for AFFR using PySpark](https://www.researchgate.net/publication/367070089_Sentiment_Classification_for_Amazon_Fine_Foods_Reviews_Using_Pyspark)  

## Hiệu suất mô hình
| Thuật toán        | Accuracy | Precision  | Recall | F1-Score  |   
|------------------|----------|--------|  ----------|--------| 
| Logistic Regression | 87.61 | 86.48 | 88.21 | 87.23 |  
| Naive Bayes          | 84.92 | 81.27 | 83.92 | 82.67 |

## Kết quả chính
- Sử dụng PySpark để huấn luyện mô hình phân loại sentiment.  
- Dữ liệu được tiền xử lý: loại bỏ trung lặp (3), loại bỏ các cột không cần thiết
- Áp dụng nhiều thuật toán ML như Logistic Regression, Decision Tree, Random Forest, Naive Bayes.  
- Sử dụng BoW và TF-IDF để trích xuất đặc trưng
- Kết quả với độ chính xác cao, khả năng của LR tốt hơn NB 

## Hạn chế
- Chưa kiểm thử trên tập dữ liệu khác.  
- hiệu suất phụ thuộc vào Spark cluster
