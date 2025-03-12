# User Sentiment Analysis on Product Reviews Using RoBERTa  

## 1. Introduction  
Trong thời đại thương mại điện tử đang phát triển nhanh chóng, đánh giá sản phẩm của người dùng đóng vai trò vô cùng quan trọng trong quyết định mua hàng. Phân tích cảm xúc từ đánh giá có thể giúp doanh nghiệp hiểu rõ hơn về phản hồi của khách hàng.  
Nghiên cứu của nhóm sử dụng **RoBERTa** để phân tích cảm xúc từ tập dữ liệu **Amazon Fine Food Reviews**, giúp đem lại những thông tin cần thiết về nhu cầu, xu hướng của người dùng trong lĩnh vực thực phẩm.  

## 2. Problem Statement  
Các mô hình truyền thống như Naive Bayes, SVM có hiệu suất hạn chế khi xử lý văn bản dài và ngữ cảnh phức tạp. Việc áp dụng RoBERTa, một mô hình Transformer tiên tiến, có thể cải thiện độ chính xác nhưng cần được đánh giá kỹ lưỡng trên dữ liệu thực tế.  

## 3. Objectives  
- Áp dụng RoBERTa vào bài toán phân tích cảm xúc.  
- So sánh hiệu suất của RoBERTa với các mô hình truyền thống.  
- Xác định các yếu tố quan trọng trong phản hồi của khách hàng.  

## 4. Methodology  
- **Dữ liệu:** [Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) từ Kaggle.  
- **Mô hình:** RoBERTa  
- **Công cụ:** Python, Hugging Face Transformers, NLTK, Scikit-learn.  
- **Đánh giá:** Sử dụng **Accuracy, F1-score** để so sánh hiệu suất mô hình.  

## 5. Expected Outcomes  
- Mô hình RoBERTa dự đoán có thể đạt độ chính xác trên 75% trên tập dữ liệu.  
- Phân tích ưu nhược điểm của RoBERTa trong bài toán phân tích cảm xúc.

## 6. Planning  
| **Giai đoạn** | **Nội dung** | **Thời gian** |
|--------------|-------------|--------------|
| 1 | Nghiên cứu lý thuyết về RoBERTa | 1 tuần |
| 2 | Thu thập, xử lý dữ liệu | 1 tuần |
| 3 | Xây dựng, huấn luyện mô hình | 2 tuần |
| 4 | Đánh giá, so sánh với các mô hình khác | 1 tuần |
| 5 | Phân tích kết quả, viết báo cáo và hoàn thiện | 1 tuần |

## 7. Resources & Budget  
- **Nhân lực:** 4 sinh viên thực hiện.  
- **Công cụ:** Python, Kaggle, Google Scholar, Zotero.  

## 8. Conclusion  
- Kết quả nghiên cứu giúp các doanh nghiệp thương mại điện tử hiểu rõ hơn về **xu hướng** và **nhu cầu của khách hàng**, từ đó cải thiện trải nghiệm người dùng.  
- Mô hình có thể **mở rộng** để áp dụng cho các ngành hàng khác.  

## References  
_(Thêm danh sách tài liệu tham khảo tại đây)_
