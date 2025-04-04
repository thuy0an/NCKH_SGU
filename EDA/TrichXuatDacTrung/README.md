# Trích Xuất Đặc Trưng
Đây là trích xuất đặc trưng cho 7500 mẫu được lấy từ tập dataset.

## Giới Thiệu
Thực hiện tiền xử lý và trích xuất đặc trưng từ 7500 mẫu đánh giá được lấy từ tập dataset gốc. Các mẫu được lấy bao gồm:
- 1500 mẫu với điểm đánh giá 1 sao
- 500 mẫu với điểm đánh giá 2 sao
- 500 mẫu với điểm đánh giá 3 sao
- 500 mẫu với điểm đánh giá 4 sao
- 1500 mẫu với điểm đánh giá 5 sao

## Quy Trình Xử Lý

### 1. Tiền xử lý dữ liệu:
Chuyển văn bản thành chữ thường
Loại bỏ các thẻ HTML
Xóa dấu câu
Tách từ (tokenization)
Loại bỏ stopwords
Lemmatization (chuyển từ về dạng gốc)

### 2. Trích xuất đặc trưng:
Đặc trưng cơ bản: độ dài văn bản, số lượng từ
Đặc trưng Bag of Words (BoW)
Phân tích tần suất từ
