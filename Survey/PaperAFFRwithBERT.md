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
- Dùng fine-tuning BERT để phân loại sentiment.  
- Chia dataset thành ba nhóm: training groups, validating groups, testing groups (7:2:1)
- Tập trung vào xử lý các thông tin với text là English
- Xử lý dữ liệu, bỏ giá trị trùng lặp.  
- Chưa kiểm tra trên tập dữ liệu khác.

## Hạn chế
- Không có so sánh với mô hình khác.
