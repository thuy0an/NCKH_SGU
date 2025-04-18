# Chapter 6: Understand Your Data With Visualization

You must understand your data in order to get the best results from machine learning algorithms. The fastest way to learn more about your data is to use data visualization. In this chapter, you will discover exactly how you can visualize your machine learning data in Python using Pandas. Recipes in this chapter use the Pima Indians onset of diabetes dataset introduced in Chapter 4. Let’s get started.

---

## 6.1 Univariate Plots (Biểu đồ Đơn Biến)

Phần này tập trung vào cách sử dụng các biểu đồ để hiểu rõ hơn từng thuộc tính của dữ liệu một cách độc lập, tức là xem xét sự phân phối và đặc điểm của từng biến mà không phụ thuộc vào các biến khác.

### 1. Histograms (Biểu đồ tần suất)

Biểu đồ tần suất là công cụ nhanh chóng giúp bạn hiểu được sự phân phối của một thuộc tính. Dữ liệu được chia thành các khoảng (bins), và mỗi khoảng này sẽ thể hiện số lượng quan sát trong đó.
- **Lợi ích**: Bạn có thể nhận diện được phân phối của dữ liệu (ví dụ: phân phối chuẩn Gaussian, phân phối lệch hoặc phân phối mũ).
- **Công dụng**: Giúp phát hiện các điểm ngoại lệ (outliers) và hiểu rõ hơn về sự phân bố của dữ liệu.

### 2. Density Plots (Biểu đồ mật độ)

Biểu đồ mật độ là một dạng biểu đồ tần suất nhưng thay vì vẽ các cột, ta sẽ có một đường cong mượt mà thể hiện sự phân bố dữ liệu. Điều này giúp bạn dễ dàng nhận diện các xu hướng trong dữ liệu.
- **Lợi ích**: Cung cấp một cái nhìn trực quan hơn về phân phối của dữ liệu.
- **Công dụng**: Giúp nhận diện các thuộc tính có phân phối gần giống nhau hoặc các điểm bất thường trong dữ liệu.

### 3. Box and Whisker Plots (Biểu đồ hộp và râu)

Biểu đồ hộp và râu tóm tắt phân phối của dữ liệu bằng cách vẽ một hình hộp (box) xung quanh 50% dữ liệu nằm giữa phân vị 25 và phân vị 75, và các "râu" thể hiện sự phân bố rộng hơn. Dưới và trên hộp là các điểm ngoại lệ.
- **Lợi ích**: Giúp nhanh chóng nhận diện các phần trăm của dữ liệu (medians, quartiles) và phát hiện các giá trị ngoại lệ.
- **Công dụng**: Tóm tắt sự phân bố và sự spread (tỏa rộng) của dữ liệu.

---

## 6.2 Multivariate Plots (Biểu đồ Đa Biến)

Phần này cung cấp các phương pháp để trực quan hóa mối quan hệ giữa nhiều thuộc tính (biến) trong dữ liệu của bạn. Điều này hữu ích khi bạn muốn hiểu rõ hơn về mối quan hệ giữa các thuộc tính và xem xét ảnh hưởng qua lại giữa chúng.

### 1. Correlation Matrix Plot (Biểu đồ ma trận tương quan)

Tương quan giữa hai biến cho biết mức độ liên hệ giữa chúng. Nếu hai biến thay đổi theo cùng một hướng, chúng có tương quan dương, nếu thay đổi ngược chiều thì tương quan âm. Một ma trận tương quan cho phép bạn tính toán mức độ tương quan giữa tất cả các cặp biến trong dữ liệu và trực quan hóa nó.
- **Lợi ích**: Bạn có thể nhận diện được những thuộc tính có sự tương quan mạnh mẽ với nhau, từ đó giúp bạn quyết định biến nào có thể bị loại bỏ hoặc sử dụng chung.
- **Công dụng**: Giúp nhận diện sự đa cộng tuyến (multicollinearity), vấn đề mà một số thuật toán máy học có thể gặp phải khi có các biến đầu vào có sự tương quan mạnh.

### 2. Scatter Plot Matrix (Ma trận biểu đồ phân tán)

Biểu đồ phân tán giúp hiển thị mối quan hệ giữa hai biến dưới dạng các điểm trong không gian 2 chiều. Một ma trận biểu đồ phân tán là sự kết hợp của tất cả các biểu đồ phân tán giữa các cặp thuộc tính trong dữ liệu. Điều này giúp bạn tìm hiểu mối quan hệ giữa tất cả các cặp thuộc tính và nhận diện các mối quan hệ có cấu trúc giữa chúng.
- **Lợi ích**: Bạn có thể thấy rõ các mối quan hệ tuyến tính hoặc phi tuyến giữa các thuộc tính.
- **Công dụng**: Giúp tìm ra các biến có mối quan hệ mạnh mẽ hoặc có thể có khả năng dự đoán cao.

---

## 6.3 Summary (Tóm Tắt)

Trong chương này, bạn đã khám phá một số cách để hiểu rõ dữ liệu của mình thông qua các biểu đồ trong Python sử dụng thư viện Pandas. Những biểu đồ này giúp bạn nhận diện các phân phối đơn biến, mối quan hệ giữa các biến, và các đặc điểm khác trong dữ liệu, từ đó cung cấp thông tin hữu ích cho quá trình chuẩn bị dữ liệu cho các mô hình máy học.

**Các phương pháp chính bao gồm**:
- **Biểu đồ tần suất** (Histograms)
- **Biểu đồ mật độ** (Density Plots)
- **Biểu đồ hộp và râu** (Box and Whisker Plots)
- **Biểu đồ ma trận tương quan** (Correlation Matrix Plot)
- **Ma trận biểu đồ phân tán** (Scatter Plot Matrix)
