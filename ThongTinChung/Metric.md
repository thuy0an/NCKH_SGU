**Độ đo (metric)** là tiêu chí, chỉ số để đánh giá hiệu suất của mô hình, phương pháp được dùng trong bài báo.  

### **Trong bài toán phân loại (Classification)**  
- **Accuracy**: Độ chính xác, tỉ lệ dự đoán đúng trên tổng số mẫu.  
- **Loss**: Hàm mất mát, **cross-entropy loss**, đo sự khác biệt giữa dự đoán của mô hình và giá trị thực tế (**ground truth**).  
- **Precision**: Độ chính xác theo lớp dương, số dự đoán là tích cực bao nhiêu là đúng.  
- **Recall**: Khả năng phát hiện mẫu dương, trong số tất cả mẫu thực sự là tích cực, mô hình dự đoán đúng bao nhiêu?  
- **F1-score**: Trung bình hài hòa giữa **Precision** và **Recall**.  
- **AUC-ROC**: Đánh giá khả năng mô hình phân biệt giữa hai lớp.  

### **Trong bài toán hồi quy (Regression)**  
- **RMSE (Root Mean Squared Error)**: Lỗi bình phương trung bình gốc.  
- **MAE (Mean Absolute Error)**: Sai số trung bình tuyệt đối.  
- **MAPE (Mean Absolute Percentage Error)**: Lỗi phần trăm trung bình tuyệt đối.  
- **R^2 (Hệ số xác định)**: Đo lường mức độ phù hợp của mô hình với dữ liệu.  

### **Trong xử lý ngôn ngữ tự nhiên (NLP)**  
- **BLEU Score**: Đánh giá chất lượng của mô hình dịch máy.  
- **ROUGE Score**: Đánh giá độ chính xác của mô hình tóm tắt văn bản.  
- **Perplexity**: Đánh giá chất lượng của mô hình sinh văn bản.  
