# Chương 10: Các Chỉ Số Đánh Giá Hiệu Suất Thuật Toán Máy Học

Các chỉ số đo lường hiệu suất của thuật toán máy học rất quan trọng. Việc lựa chọn các chỉ số này ảnh hưởng đến cách bạn đánh giá và so sánh hiệu suất của các thuật toán, cũng như quyết định thuật toán nào sẽ được chọn. Chỉ số bạn chọn cũng ảnh hưởng đến cách bạn đánh giá tầm quan trọng của các đặc tính khác nhau trong kết quả. Trong chương này, bạn sẽ tìm hiểu cách chọn và sử dụng các chỉ số hiệu suất khác nhau trong Python với scikit-learn.

## 10.1 Các Chỉ Số Đánh Giá Thuật Toán
Ở phần này, các chỉ số đánh giá thuật toán sẽ được giới thiệu cho cả bài toán phân loại và hồi quy. Dữ liệu sẽ được tải trực tiếp từ kho dữ liệu UCI Machine Learning.

- **Phân loại:** Dữ liệu "Pima Indians onset of diabetes" được sử dụng cho bài toán phân loại nhị phân.
- **Hồi quy:** Dữ liệu "Boston House Price" được sử dụng cho bài toán hồi quy.
- Các thuật toán đánh giá là Hồi quy Logistic cho phân loại và Hồi quy Tuyến tính cho bài toán hồi quy.

Chúng ta sử dụng phương pháp cross-validation 10 lần để minh họa mỗi chỉ số, vì đây là kịch bản thường gặp khi áp dụng các chỉ số đánh giá thuật toán khác nhau.

## 10.2 Các Chỉ Số Đánh Giá Phân Loại
Các bài toán phân loại là loại bài toán phổ biến nhất trong học máy, và có rất nhiều chỉ số có thể dùng để đánh giá các dự đoán cho bài toán phân loại. Trong phần này, các chỉ số sẽ được xem xét bao gồm:

- **Độ chính xác phân loại (Classification Accuracy)**
- **Mất mát logarithmic (Logarithmic Loss)**
- **Diện tích dưới đường cong ROC (Area Under ROC Curve - AUC)**
- **Ma trận nhầm lẫn (Confusion Matrix)**
- **Báo cáo phân loại (Classification Report)**

### 10.2.1 Độ Chính Xác Phân Loại (Classification Accuracy)
Đây là tỷ lệ dự đoán đúng trong tổng số dự đoán. Độ chính xác này chỉ thực sự phù hợp khi số lượng quan sát giữa các lớp là như nhau, và tất cả các lỗi dự đoán đều quan trọng như nhau. Độ chính xác được tính bằng tỷ lệ dự đoán đúng và có thể chuyển thành tỷ lệ phần trăm.

### 10.2.2 Mất Mát Logarithmic (Logarithmic Loss)
Logloss là một chỉ số hiệu suất đánh giá dự đoán xác suất cho từng lớp. Mất mát logarithmic phạt các dự đoán sai theo mức độ tự tin của dự đoán đó. Mất mát này càng nhỏ thì hiệu suất của mô hình càng tốt.

### 10.2.3 Diện Tích Dưới Đường Cong ROC (AUC)
AUC là chỉ số đo lường khả năng phân biệt giữa lớp dương tính và âm tính của mô hình. Một giá trị AUC bằng 1.0 cho thấy mô hình phân loại chính xác hoàn toàn, trong khi AUC bằng 0.5 cho thấy mô hình gần như dự đoán ngẫu nhiên.

### 10.2.4 Ma Trận Nhầm Lẫn (Confusion Matrix)
Ma trận nhầm lẫn là một công cụ hữu ích để trình bày độ chính xác của mô hình với hai lớp hoặc hơn. Ma trận này cho thấy số lượng dự đoán đúng và sai cho mỗi lớp.

### 10.2.5 Báo Cáo Phân Loại (Classification Report)
Báo cáo phân loại trong scikit-learn cung cấp các chỉ số như precision, recall, F1-score, và support cho mỗi lớp, giúp bạn nhanh chóng đánh giá hiệu suất của mô hình phân loại.

## 10.3 Các Chỉ Số Đánh Giá Hồi Quy
Trong phần này, chúng ta sẽ xem xét ba chỉ số phổ biến để đánh giá các dự đoán cho bài toán hồi quy:

- **Lỗi tuyệt đối trung bình (Mean Absolute Error - MAE)**
- **Lỗi bình phương trung bình (Mean Squared Error - MSE)**
- **Chỉ số R²**

### 10.3.1 Lỗi Tuyệt Đối Trung Bình (Mean Absolute Error - MAE)
MAE là tổng của các sai số tuyệt đối giữa các giá trị dự đoán và giá trị thực. Nó cho bạn biết độ lớn của sai số, nhưng không cho biết sai số đó có bị thừa hay thiếu.

### 10.3.2 Lỗi Bình Phương Trung Bình (Mean Squared Error - MSE)
MSE cung cấp thông tin về độ lớn của sai số, nhưng vì bình phương các sai số, MSE có thể nhạy cảm hơn với các sai số lớn. Để trả lại đơn vị ban đầu, có thể tính toán căn bậc hai của MSE, tạo ra chỉ số RMSE (Root Mean Squared Error).

### 10.3.3 Chỉ Số R² (R Squared)
R² đo lường mức độ phù hợp của mô hình với dữ liệu thực tế. Giá trị R² càng gần 1.0, mô hình càng phù hợp với dữ liệu, trong khi giá trị gần 0 cho thấy mô hình không phù hợp.

## 10.4 Tổng Kết
Trong chương này, bạn đã tìm hiểu các chỉ số mà bạn có thể sử dụng để đánh giá các thuật toán máy học. Bạn đã học về ba chỉ số phân loại: Độ chính xác, Mất mát logarithmic và Diện tích dưới đường cong ROC. Bạn cũng đã học về hai phương pháp tiện ích cho kết quả phân loại: Ma trận nhầm lẫn và Báo cáo phân loại. Cuối cùng, bạn cũng đã học về ba chỉ số cho bài toán hồi quy: Lỗi tuyệt đối trung bình, Lỗi bình phương trung bình và R².

Chương tiếp theo sẽ giới thiệu các thuật toán học máy, bắt đầu với các kỹ thuật phân loại.
