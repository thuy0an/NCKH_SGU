### **Các thuật ngữ gặp trong việc tìm hiểu đề tài**

### **Classification**
**Classification (Bài toán phân lớp)**: Bài toán phân lớp là quá trình phân lớp một đối tượng dữ liệu vào một hay nhiều lớp đã cho trước nhờ một mô hình phân lớp

### **Sentiment Analysis**
**Sentiment Analysis (Phân tích cảm xúc)**: Phân tích cảm xúc là quy trình mà dữ liệu được trích xuất từ ý kiến, đánh giá và cảm xúc của con người từ **văn bản** liên quan đến các thực thể, sự kiện. Trong quá trình ra quyết định, ý kiến của người khác có ảnh hưởng đến sự tiện lợi của khách hàng, giúp họ đưa ra các quyết định liên quan đến mua sắm trực tuyến, chọn sự kiện, sản phẩm hoặc thực thể. Nhiệm vụ của bài toán phân lớp là cần tìm một mô hình phần lớp để khi có dữ liệu mới thì có thể xác định được dữ liệu đó thuộc vào phân lớp nào.

### **NLP (Natural Language Processing)**
**NLP (xử lý ngôn ngữ tự nhiên)**: lĩnh vực cảu AI kết hợp giữa ngôn ngữ học và học máy để giúp máy tính hiểu, phân tích, tạo ra và phản hồi ngôn ngữ con người một cách tự động

### **BoW (Bag of Words)**
**BoW (túi từ)**: là một kỹ thuật trong xử lý ngôn ngữ tự nhiên (NLP) để biểu diễn văn bản dưới dạng một tập hợp các từ mà không quan tâm đến ngữ pháp hoặc vị trí của từ trong câu.

### **Transformer**
**Transformer**: kiến trúc mạng nơ-ron, được thiết kế để xử lý dữ liệu chuỗi, đặc biệt là xử lý ngôn ngữ tự nhiên.

**Cách hoạt động**
VD có 2 câu: "I love  this movie" và "This movie is amazing"
- Bước 1: xây dựng danh sách từ vựng
Chuỗi sẽ được tạo thành danh sách các từ xuất hiện (vocabulary)
["I", "love", "this", "movie", "is", "amazing"]
- Bước 2: Biểu diễn văn bản dưới dạng vector
Chuyển mỗi câu thành một vector đếm số lần xuất hiện của từ trong câu đó:

| Câu                  | I | love | this | movie | is | amazing |
|----------------------|---|------|------|-------|----|---------|
| "I love this movie" | 1 | 1    | 1    | 1     | 0  | 0       |
| "This movie is amazing" | 0 | 0    | 1    | 1     | 1  | 1       |

### ** Các phương pháp phân tích cảm xúc**
**Chia thành 3 nhóm**
-Mô hình dựa trên quy tắc: dùng từ điển cảm xúc
-Mô hình machine learning: dùng thuật toán ML để học dữ liệu
-Mô hình deep learning: dùng mạng neuron sâu


### **Mô hình Deep Learning**

### **BERT (Bidirectional Encoder Representations from Transformers)**
**BERT**: là mô hình NLP dựa trên transformer, học theo cách hai chiều để hiểu ngữ cảnh

### **RoBERTa (Robustly Optimized BERT Pretraining Approach)**
**RoBERTa**: RoBERTa là một phiên bản cải tiến của BERT, được huấn luyện kỹ hơn bằng cách sử dụng nhiều dữ liệu hơn, loại bỏ tác vụ NSP (Next Sentence Prediction).

### **Mô hình Machine Learning**

### **SVM (Support Vector Machine)**
**SVM**: thuật toán phân loại mạnh, tìm đường phân tách tối ưu giữa các nhóm dữ liệu

### **Naive Bayes**
**Naive Bayes**: Mô hình thống kê dựa trên Định lý Bayes, giả định các từ trong văn bản độc lập nhau.

### **Logistic Regression**
**Logistic Regression**: Một thuật toán Machine Learning tuyến tính dùng để phân loại nhị phân hoặc đa lớp.


### **Mô hình dựa trên quy tắc**

### **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
**VADER**: VADER phân tích cảm xúc dựa trên từ điển, tối ưu cho dữ liệu mạng xã hội.


