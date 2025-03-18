# AFFR with BERT Model

## Thông tin bài báo
- **Tác giả**: Xinyue Zhao, Yuandong Sun  
- **Năm**: 2022  
- **Nguồn**: Columbia University, Los Angeles  
- **Mô hình**: BERT  
- **Dataset**: Amazon Fine Food Reviews  
- **Link**: [Amazon Fine Food Reviews with BERT Model](https://www.sciencedirect.com/science/article/pii/S1877050922014971)
## Hiệu suất mô hình
- **Accuracy**: 79.82%  
- **Loss**: 0.5433  

## Kết quả chính
- Trích xuất đặc trưng bằng Word cloud, xử lý các dữ liệu thừa và loại bỏ các dữ liệu không cần thiết
- Tạo cột mới là Combo (kết hợp giữa summary và texxt)
- Tập trung vào xử lý các thông tin với text là English
- Chia dataset thành ba nhóm: training groups, validating groups, testing groups (7:2:1)
- Dùng fine-tuning BERT để phân loại sentiment.  

## Hạn chế
- Không có so sánh với mô hình khác.
- - Chưa kiểm tra trên tập dữ liệu khác.
