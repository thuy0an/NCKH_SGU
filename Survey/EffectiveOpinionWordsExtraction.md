# Effective Opinion Words Extraction for Food Reviews Classification

##  Thông tin bài báo  
- **Tác giả**: Phuc Quang Tran, Ngoan Thanh Trieu, Nguyen Vu Dao, Hai Thanh Nguyen, Hiep Xuan Huynh  
- **Năm**: 2020  
- **Nguồn**: International Journal of Advanced Computer Science and Applications (IJACSA)  
- **Mô hình**: Decision Tree, Random Forest, Gradient Boosting Classifier  
- **Dataset**: Amazon Fine Food Reviews  
- **Link**: [Effective Opinion Words Extraction for Food Reviews Classification
](https://www.researchgate.net/publication/343425341_Effective_Opinion_Words_Extraction_for_Food_Reviews_Classification)

##  Hiệu suất mô hình  
| Mô hình | Accuracy |
|---------|----------|
| Decision Tree (DTC) | 75.2% |
| Random Forest (RF) | 82.1% |
| Gradient Boosting Classifier (GBC) | 84.5% |

##  Kết quả chính  
- Dùng phương pháp trích xuất từ ngữ mang cảm xúc (Opinion Words Extraction - OWE) để cải thiện độ chính xác.  
- Trích xuất đặc trưng bằng Doc2Vec (PV-DM, PV-DBOW) để biểu diễn dữ liệu. 
- Chia tập dữ liệu training 14825 và testing 3707
- Huấn luyện mô hình Machine Learning:  
  - Gradient Boosting Classifier (GBC) đạt độ chính xác cao nhất **84.5%**.  
  - Random Forest (RF) đạt 82.1%, hiệu suất tốt nhưng thấp hơn GBC.  
  - Decision Tree (DTC) có độ chính xác thấp nhất (75.2%), dễ bị overfitting.
