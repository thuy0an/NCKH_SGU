# User Sentiment Analysis on Product Reviews Using Machine Learning    

## 1. Introduction  
Trong thời đại thương mại điện tử đang phát triển nhanh chóng, đánh giá sản phẩm của người dùng đóng vai trò vô cùng quan trọng trong quyết định mua hàng. Phân tích cảm xúc từ đánh giá có thể giúp doanh nghiệp hiểu rõ hơn về phản hồi của khách hàng.  
Nghiên cứu của nhóm áp dụng và so sánh các  **thuật toán Machine Learning (ML)** trong bài toán phân tích cảm xúc trên tập dữ liệu **Amazon Fine Food Reviews**. Nóm nghiên cứu sẽ đánh giá hiệu suất của các mô hình để tìm ra phương pháp hiệu quả nhất để phân loại cảm xúc.  

## 2. Problem Statement  
Các thuật toán Machine Learning khác nhau có hiệu suất khác nhau khi áp dụng cho bài toán phân tích cảm xúc. Một số mô hình đơn giản như Naive Bayes có thể hoạt động tốt với dữ liệu văn bản ngắn nhưng kém hiệu quả với các đánh giá dài và phức tạp. Trong khi đó, SVM hay Random Forest có thể cải thiện độ chính xác nhưng yêu cầu nhiều tài nguyên tính toán hơn.  

## 3. Objectives  
- Triển khai mô hình và so sánh hiệu suất giữa các thuật toán ML  
- Đánh giá độ chính xác, khả năng của từng mô hình.  
- Xác định phương pháp tôi ưu để áp dụng vào bài toán thực tế.   

## 4. Methodology  
- **Dữ liệu:** [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) từ Kaggle.  
- **Nhãn cảm xúc:** tích cực (positive), tiêu cực (negative), trung lập (neutral)
- **Tiền xử lý:** làm sạch dự liệu các dấu câu, kí tự đặc biệt, vector hóa văn bản(BoW)
- **Công cụ:** Python, NLTK, Scikit-learn, Seaborn.  
- **Đánh giá:** Sử dụng **Accuracy, Percision, Recall, F1-score** để so sánh hiệu suất mô hình.  

## 5. Expected Outcomes  
- Đánh giá được mô hình thuật toán có tốc độ nhanh và kết quả tốt.    
- Chỉ ra được mô hình nào phù hợp nhất để có thể triển khai.  

## 6. Planning  
| **Giai đoạn** | **Nội dung** | **Thời gian** |
|--------------|-------------|--------------|
| 1 | Nghiên cứu tài liệu các thuật toán ML | 2 tuần |
| 2 | Thu thập, xử lý dữ liệu | 1 tuần |
| 3 | Triển khai huấn luyện các mô hình ML | 2 tuần |
| 4 | Đánh giá, so sánh kết quả | 1 tuần |
| 5 | Phân tích kết quả, viết báo cáo và hoàn thiện | 1 tuần |

## 7. Resources & Budget  
- **Nhân lực:** 4 sinh viên thực hiện.  
- **Công cụ:** Python, Kaggle, Google Scholar, Zotero.  

## 8. Conclusion  
- Kết quả nghiên cứu giúp các doanh nghiệp thương mại điện tử hiểu rõ hơn về **xu hướng** và **nhu cầu của khách hàng**, từ đó cải thiện trải nghiệm người dùng.
- Xác định thuật toán ML tốt nhất cho bài toán phân tích cảm xúc từ đánh giá sản phẩm.  

## References  
_(Thêm danh sách tài liệu tham khảo tại đây)_
