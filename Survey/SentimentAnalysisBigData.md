
# Sentiment Analysis of Online Food Reviews using Big Data Analytics

## Thông tin bài báo
- **Tác giả:** Hafiz Muhammad Ahmed, Mazhar Javed Awan, Nabeel Sabir Khan, Awais Yasin, Hafiz Muhammad Faisal Shehzad
- **Năm:** 2021
- **Nguồn:** University of Management and Technology, Lahore, Pakistan
- **Mô hình:** Linear SVC, Logistic Regression, Naïve Bayes
- **Dataset:** Amazon Fine Food Reviews
- **Link:** [Sentiment Analysis of Online Food Reviews using Big Data Analytics](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3827110)

## Hiệu suất mô hình
| Thuật toán        | Accuracy | Precision  | Recall | F1-Score  |   
|------------------|----------|--------|  ----------|--------| 
| Logistic Regression | 87.38 | 86.54 | 88.78 | 87.64 |  
| Naive Bayes          | 83.43 | [82.35] | [88.78] | [85.54] |
| LinearSVC         | 88.38 | [88.54] | [88.39] | [88.46] |

## Kết quả chính
- Sử dụng Apache Spark để xử lý dữ liệu lớn trong phân tích cảm xúc.
- Linear SVC có độ chính xác cao nhất (> 88%).
- Hệ thống có thể phân loại đánh giá thành **tích cực** hoặc **tiêu cực** với hiệu suất tối ưu.

## Hạn chế
- Chưa thử nghiệm với các mô hình học sâu như BERT.
- Cần cải thiện hiệu suất trên tập dữ liệu lớn hơn.
