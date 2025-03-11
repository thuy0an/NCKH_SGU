Độ đo (metric) là tiêu chí, chỉ số để đánh giá hiệu suất của mô hình, phương pháp được dùng trong bài báo
**Trong bài toán phân loại classification**
Accuracy: độ chính xác, tỉ lệ dự đoán đúng trên tổng số mẫu
Loss: hàm mất mát, cross-entropy loss, đo sự khác biệt giữa dự đoán của mô hình và giá trị thực tế (ground truth)
Percision: độ chính xác theo lớp dương, số dự đoán là tích cực bao nhiêu là đúng
Recall: khả năng phát hiện mẫu dương, trong số tất cả mẫu thực sự là tích cực, mô hình dự đoán đúng bao nhiêu?
F1-score: trung bình hài hòa giữa Precision và Recall
AUC-ROC: đánh giá khả năng mô hình phân biệt giữa hai lớp
**Trong bài toán hồi quy regression**
RMSE (root mean squared error): lỗi bình phương trung bình gốc
MAE (Mean absolute error): sai số trung bình tuyệt đối
MAPE (Mean absolute percentage error): lỗi phần trăm trung bình tuyệt đối
R^2 (hệ số xác định): đo lường mức độ phù hợp của mô hình với dữ liệu
**Trong xử lý ngôn ngữ tự nhiên NLP**
BLEU Score: cho dịch máy
ROUGE Score: cho tóm tắt văn bản
Perplexity: cho mô hình sinh văn bản
