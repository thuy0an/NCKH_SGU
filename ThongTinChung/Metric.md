**Độ đo (metric)** là tiêu chí, chỉ số để đánh giá hiệu suất của mô hình, phương pháp được dùng trong bài báo.  

### 1. Metrics trong bài toán phân loại (Classification)  
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


## 2. Confusion Matrix (Ma trận nhầm lẫn)

Bảng này giúp chúng ta hiểu cách mô hình phân loại đúng hay sai. Là một phương pháp đánh giá kết quả của những bài toán phân loại với việc xem xét cả những chỉ số về độ chính xác và độ bao quát của các dự đoán cho từng lớp.

|                 | Dự đoán Tích cực (1) | Dự đoán Tiêu cực (0) |
|---------------|--------------------|--------------------|
| **Positive (1)** | True Positive (TP)  | False Negative (FN)  |
| **Negative (0)** | False Positive (FP)  | True Negative (TN)  |

- **TP (True Positive)**: Dự đoán đúng bài đánh giá là tích cực.
- **TN (True Negative)**: Dự đoán đúng bài đánh giá là tiêu cực
- **FP (False Positive)**: Dự đoán sai, thực tế là tiêu cực nhưng mô hình dự đoán là tích cực.
- **FN (False Negative)**: Dự đoán sai, thực tế là tích cực nhưng mô hình dự đoán là tiêu cực.


## 3. Công thức tính các metric

- **Accuracy (Độ chính xác)**  
  $$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$

- **Precision (Độ chính xác theo lớp dương)**  
  $$ Precision = \frac{TP}{TP + FP} $$

- **Recall (Khả năng phát hiện mẫu dương - Độ phủ)**  
  $$ Recall = \frac{TP}{TP + FN} $$

- **F1-score (Trung bình hài hòa giữa Precision và Recall)**  
  $$ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

- **AUC-ROC (Đánh giá khả năng phân biệt giữa hai lớp)**  
  - **True Positive Rate (TPR - Sensitivity/Recall)**  
    $$ TPR = \frac{TP}{TP + FN} $$
  - **False Positive Rate (FPR - 1-Specificity)**  
    $$ FPR = \frac{FP}{FP + TN} $$


