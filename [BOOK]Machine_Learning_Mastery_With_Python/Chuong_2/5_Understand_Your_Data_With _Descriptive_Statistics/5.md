# Chapter 5 - Understand Your Data With Descriptive Statistics

## 5.1 Peek at Your Data
Khi bắt đầu với một tập dữ liệu, không có cách nào thay thế việc "nhìn vào" dữ liệu thô. Việc này giúp bạn phát hiện ra những điều mà bạn không thể nhận thấy thông qua các phân tích thống kê hoặc thuật toán.

- **Mục tiêu**: Cung cấp cái nhìn trực quan về dữ liệu, đặc biệt là để nhận diện nhanh chóng các mẫu hoặc bất thường trong dữ liệu.
- **Tại sao quan trọng**: Việc kiểm tra dữ liệu đầu tiên giúp bạn có cái nhìn tổng quan về bộ dữ liệu và hiểu được cấu trúc của nó trước khi thực hiện các phân tích phức tạp hơn.

## 5.2 Dimensions of Your Data
Khi làm việc với dữ liệu, bạn cần hiểu rõ số lượng dữ liệu bạn có. Điều này bao gồm số lượng hàng (instances) và cột (attributes).

- **Mục tiêu**: Hiểu rõ về kích thước của dữ liệu để có kế hoạch xử lý phù hợp.
- **Tại sao quan trọng**: Việc biết được kích thước của dữ liệu giúp bạn quyết định xem có đủ dữ liệu để huấn luyện mô hình hay không, và liệu bạn có cần giảm bớt số lượng đặc trưng hay không.

## 5.3 Data Type For Each Attribute
Kiểu dữ liệu của từng thuộc tính là một yếu tố quan trọng trong phân tích dữ liệu. Dữ liệu có thể là chuỗi, số nguyên, hoặc số thực.

- **Mục tiêu**: Kiểm tra kiểu dữ liệu của các thuộc tính trong dữ liệu.
- **Tại sao quan trọng**: Việc hiểu kiểu dữ liệu giúp bạn có thể quyết định liệu có cần chuyển đổi dữ liệu thành một kiểu khác hay không.

## 5.4 Descriptive Statistics
Các thống kê mô tả là các công cụ mạnh mẽ giúp bạn hiểu về phân phối và đặc điểm của từng thuộc tính trong bộ dữ liệu.

- **Mục tiêu**: Tóm tắt dữ liệu thông qua các chỉ số thống kê cơ bản như trung bình, độ lệch chuẩn, giá trị cực tiểu và cực đại.
- **Tại sao quan trọng**: Các thống kê mô tả cung cấp cái nhìn tổng quan về sự phân phối của dữ liệu, từ đó giúp bạn nhận diện các vấn đề như dữ liệu không đồng đều hoặc các giá trị ngoại lệ.

## 5.5 Class Distribution (Classification Only)
Khi làm việc với bài toán phân loại, việc kiểm tra sự phân bố của các lớp là rất quan trọng.

- **Mục tiêu**: Kiểm tra sự phân bố của các lớp trong dữ liệu phân loại.
- **Tại sao quan trọng**: Nếu các lớp bị mất cân đối, mô hình có thể gặp khó khăn trong việc phân loại chính xác các lớp ít gặp hơn, do đó bạn có thể cần các kỹ thuật xử lý đặc biệt.

## 5.6 Correlations Between Attributes
Sự tương quan giữa các thuộc tính cho biết mức độ liên hệ giữa hai hoặc nhiều thuộc tính trong dữ liệu.

- **Mục tiêu**: Xác định các thuộc tính có mối quan hệ tương quan mạnh với nhau.
- **Tại sao quan trọng**: Nếu hai hoặc nhiều thuộc tính có mối quan hệ tương quan mạnh, bạn có thể cân nhắc loại bỏ hoặc kết hợp chúng để tránh đa cộng tuyến và cải thiện hiệu suất mô hình.

## 5.7 Skew of Univariate Distributions
Skewness (độ lệch) cho biết mức độ lệch của phân phối dữ liệu so với phân phối chuẩn (Gaussian).

- **Mục tiêu**: Kiểm tra độ lệch của các thuộc tính trong dữ liệu.
- **Tại sao quan trọng**: Nếu dữ liệu có độ lệch cao, bạn có thể cần phải chuyển đổi hoặc chuẩn hóa dữ liệu để cải thiện hiệu suất của mô hình.

## 5.8 Tips To Remember
Một số lời khuyên quan trọng khi sử dụng thống kê mô tả để phân tích dữ liệu:
- **Xem xét kỹ các số liệu**: Việc chỉ tạo ra các thống kê mô tả là chưa đủ. Hãy dành thời gian để xem xét và suy ngẫm về những con số mà bạn thấy.
- **Hỏi tại sao**: Đặt câu hỏi về những con số này và suy nghĩ về ý nghĩa của chúng đối với vấn đề bạn đang giải quyết.
- **Ghi chú ý tưởng**: Ghi lại các quan sát và ý tưởng để có thể tham khảo lại khi cần.

## 5.9 Summary
Chương này cung cấp các phương pháp quan trọng để hiểu rõ dữ liệu trước khi bắt đầu xây dựng mô hình học máy. Những phương pháp này bao gồm việc xem qua dữ liệu, kiểm tra kích thước dữ liệu, xác định kiểu dữ liệu, phân tích thống kê mô tả, kiểm tra phân bố lớp, kiểm tra tương quan giữa các thuộc tính và đánh giá độ lệch của các phân phối.
