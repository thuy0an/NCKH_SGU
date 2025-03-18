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
<img src="https://latex.codecogs.com/svg.latex?Accuracy%20%3D%20%5Cfrac%7BTP%20&plus;%20TN%7D%7BTP%20&plus;%20TN%20&plus;%20FP%20&plus;%20FN%7D" />

- **Precision (Độ chính xác theo lớp dương)**  
<img src="https://latex.codecogs.com/svg.latex?Precision%20%3D%20%5Cfrac%7BTP%7D%7BTP%20&plus;%20FP%7D" />

- **Recall (Khả năng phát hiện mẫu dương - Độ phủ)**  
<img src="https://latex.codecogs.com/svg.latex?Recall%20%3D%20%5Cfrac%7BTP%7D%7BTP%20&plus;%20FN%7D" />

- **F1-score (Trung bình hài hòa giữa Precision và Recall)**  
<img src="https://latex.codecogs.com/svg.latex?F1%20%3D%202%20%5Ctimes%20%5Cfrac%7BPrecision%20%5Ctimes%20Recall%7D%7BPrecision%20&plus;%20Recall%7D" />


- **AUC-ROC (Đánh giá khả năng phân biệt giữa hai lớp)**  
  - **True Positive Rate (TPR - Sensitivity/Recall)**  
<img src="https://latex.codecogs.com/svg.latex?TPR%20%3D%20%5Cfrac%7BTP%7D%7BTP%20&plus;%20FN%7D" />

  - **False Positive Rate (FPR - 1-Specificity)**
  <img src="https://latex.codecogs.com/svg.latex?FPR%20%3D%20%5Cfrac%7BFP%7D%7BFP%20&plus;%20TN%7D" />

- **Loss Function (Hàm mất mát)**  
<img src="https://latex.codecogs.com/svg.latex?L%20%3D%20-%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5Em%20%5B%20y_i%20log%28%5Chat%7By%7D_i%29%20&plus;%20%281-y_i%29%20log%281-%5Chat%7By%7D_i%29%20%5D" />





