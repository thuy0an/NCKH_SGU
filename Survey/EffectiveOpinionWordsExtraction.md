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
 

--


**Tóm tắt (Abstract)**

Bài báo này tập trung vào lĩnh vực **khai phá quan điểm (opinion mining)**, còn được gọi là **phân tích tình cảm (sentiment analysis)** hoặc trí tuệ cảm xúc nhân tạo, vốn đóng vai trò quan trọng trong thương mại điện tử và nhiều lĩnh vực kinh doanh khác. Nó sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên, phân tích văn bản, ngôn ngữ học tính toán và sinh trắc học để hiểu được cảm nhận của mọi người về thương hiệu, sản phẩm hoặc dịch vụ.

Trong nghiên cứu này, các tác giả điều tra các đánh giá từ bộ dữ liệu **Amazon Fine Food Reviews** (chứa khoảng 500.000 đánh giá). Họ đề xuất một phương pháp để biến đổi các đánh giá thành các **đặc trưng (features)** bao gồm các **Từ ngữ Quan điểm (Opinion Words)**. Các đặc trưng này sau đó có thể được sử dụng cho nhiệm vụ **phân loại đánh giá (reviews classification)** bằng các thuật toán học máy. Dựa trên kết quả thu được, bài báo đánh giá những Từ ngữ Quan điểm hữu ích, có thể cung cấp thông tin để xác định xem một đánh giá là **tích cực (positive)** hay **tiêu cực (negative)**.

*   **Giải thích chi tiết:** Phần tóm tắt giới thiệu về tầm quan trọng của việc hiểu ý kiến khách hàng từ các đánh giá trực tuyến. Mục tiêu chính của bài báo là xác định những từ ngữ cụ thể (Opinion Words) trong các đánh giá thực phẩm trên Amazon, những từ này sẽ được dùng làm đầu vào cho các mô hình học máy để tự động phân loại đánh giá đó là tốt hay xấu.

---

**Từ khóa (Keywords):** Phân loại đánh giá; từ ngữ quan điểm; học máy; đặc trưng quan trọng; Amazon.

---

**I. Giới thiệu (Introduction)**

Phần giới thiệu đặt bối cảnh với sự phát triển mạnh mẽ của Internet, thương mại điện tử và mạng xã hội, nơi người dùng thường xuyên đưa ra đánh giá và quan điểm. Khai phá quan điểm (Opinion mining) trở thành công cụ hỗ trợ ra quyết định quan trọng. Mục đích chính là phân tích quan điểm, đánh giá, thái độ và cảm xúc của con người về các đối tượng khác nhau (sản phẩm, dịch vụ, sự kiện...).

Bài báo phân loại khai phá quan điểm thành 3 cấp độ:
1.  **Cấp độ tài liệu (Document-based):** Xem toàn bộ tài liệu thể hiện một quan điểm duy nhất về một đối tượng chính.
2.  **Cấp độ câu (Sentence-based):** Phân tích quan điểm trong từng câu riêng lẻ, vì một tài liệu có thể chứa nhiều quan điểm.
3.  **Cấp độ khía cạnh/đặc trưng (Aspect-based/Feature-based):** Xác định tất cả các biểu hiện cảm xúc và các khía cạnh (aspects) mà chúng đề cập đến. Cách này hữu ích khi đánh giá các thực thể có nhiều thuộc tính (ví dụ: đánh giá sản phẩm).

Các phương pháp chính để xây dựng hệ thống khai phá quan điểm bao gồm:
1.  **Dựa trên từ điển (Lexicon-based):** Sử dụng từ điển tình cảm (sentiment dictionary) và các từ ngữ đánh dấu (sentinel words) để xác định cực tính (polarity). Ưu điểm là có kiến thức rộng, nhưng hạn chế về số lượng từ và điểm số cảm xúc cố định.
2.  **Dựa trên học máy (Machine learning-based):** Sử dụng các kỹ thuật phân loại (như Naive Bayes, SVM, Cây quyết định, Mạng Nơ-ron) để học từ dữ liệu huấn luyện (đã được gán nhãn). Ưu điểm là khả năng thích ứng và tạo mô hình theo ngữ cảnh cụ thể, nhưng yêu cầu dữ liệu gán nhãn (có thể tốn kém) và khả năng dự đoán có thể không cao trên dữ liệu mới.
3.  **Kết hợp (Hybrid-based):** Kết hợp cả hai phương pháp trên để cải thiện hiệu suất.
4.  **Dựa trên học sâu (Deep learning-based):** Các phương pháp mới nổi gần đây.

Bài báo này đề xuất một phương pháp **tập trung vào việc trích xuất các Từ ngữ Quan điểm** làm đặc trưng để đưa vào các thuật toán **học máy dạng tập hợp (ensemble learning)** như Cây quyết định (Decision Tree - dtc), Gradient Boosting Classifier (gbc), và Rừng Ngẫu nhiên (Random Forest - rf) nhằm nâng cao hiệu quả phân loại.

*   **Giải thích chi tiết:** Phần giới thiệu cung cấp cái nhìn tổng quan về lĩnh vực khai phá quan điểm, các cấp độ phân tích và các hướng tiếp cận chính. Nó định vị nghiên cứu này trong bối cảnh đó, nhấn mạnh vào việc sử dụng các "Từ ngữ Quan điểm" làm đầu vào cho các mô hình học máy tập hợp để phân loại đánh giá.

---

**II. Công trình Liên quan (Related Work)**

Phần này điểm qua các nghiên cứu trước đó liên quan đến:
*   **Phân loại văn bản (Text classification) [9]:** Nhiệm vụ gán nhãn cho tài liệu dựa trên nội dung.
*   **Thuật toán phân cụm (Clustering):**
    *   **K-means [1]:** Một thuật toán phân cụm phân hoạch phổ biến.
    *   **Phân cụm phân cấp (Hierarchical Clustering) [3, 7]:** Phương pháp tổ chức dữ liệu thành các cụm có cấu trúc phân cấp. Nghiên cứu [7] kết hợp TF-IDF để chọn lọc đặc trưng trước khi phân cụm.
*   **Mô hình Bag of Words (BoW) [4]:** Biểu diễn tài liệu dựa trên tần suất xuất hiện của từ, bỏ qua thứ tự.
*   **Phân tích tình cảm sử dụng học máy [13]:** Áp dụng các thuật toán như Naive Bayes và SVM.
*   **SentiWordNet [8]:** Một tài nguyên từ vựng gán điểm tình cảm (tích cực, tiêu cực, khách quan) cho các từ trong WordNet.

*   **Giải thích chi tiết:** Phần này cho thấy các tác giả đã tham khảo các kỹ thuật nền tảng trong xử lý ngôn ngữ tự nhiên và học máy, bao gồm các cách biểu diễn văn bản (BoW, TF-IDF), phân loại, phân cụm, và các công cụ/tài nguyên phân tích tình cảm đã có.

---

**III. Phương pháp Phân loại Đánh giá Dựa trên Tập Từ ngữ Cảm xúc Đề xuất**

Phương pháp đề xuất bao gồm các thành phần chính được minh họa trong Hình 1:
1.  **Tiền xử lý (Pre-processing):** Làm sạch dữ liệu đánh giá.
2.  **Trích xuất Từ ngữ Quan điểm (Opinion Word Extraction):** Xác định các từ thể hiện cảm xúc.
3.  **Biểu diễn Vector (Vector Representation):** Sử dụng **Doc2vec** để chuyển đổi đánh giá thành vector đặc trưng.
4.  **Phân loại (Classification):** Sử dụng các thuật toán học máy tập hợp.

**(Hình 1: Sơ đồ quy trình Trích xuất Từ ngữ Quan điểm và Phân loại Đánh giá)**

*   **A. Mô tả Bộ dữ liệu (Dataset Description):**
    *   Sử dụng bộ dữ liệu Amazon Fine Food Reviews (>500.000 đánh giá).
    *   Thông tin chi tiết trong Bảng I (Số lượng đánh giá, người dùng, sản phẩm...).
    *   Giữ lại nội dung đánh giá và nhãn (tích cực/tiêu cực), loại bỏ thông tin khác và các đánh giá trùng lặp.
    *   Dữ liệu được chia thành tập huấn luyện và tập kiểm tra (Bảng II).

**(Bảng I: Chi tiết dữ liệu; Bảng II: Dữ liệu Huấn luyện và Kiểm tra)**

*   **B. Tiền xử lý (Pre-Processing):**
    *   Loại bỏ số, ký tự đặc biệt, từ dừng (stop words) - những từ phổ biến nhưng ít mang nghĩa (ví dụ: "on", "the", "at").
    *   Sử dụng công cụ **NLTK (Natural Language Toolkit)** [12] để tách từ (tokenization) và loại bỏ từ dừng.
    *   Giữ lại các từ mang nghĩa như danh từ, động từ, tính từ, trạng từ.

*   **C. Trích xuất Từ ngữ Quan điểm (Opinion Word Extraction):**
    *   Trích xuất các từ thể hiện tình cảm từ đánh giá sản phẩm (ví dụ: "good", "bad", "great", "better").
    *   Sử dụng công cụ **OpinionFinder** [14] để tự động xác định các câu chủ quan và trích xuất các từ ngữ quan điểm (thường là tính từ, trạng từ).

*   **D. Doc2vec:**
    *   Doc2vec (hay Paragraph Vector) [11] là một phương pháp để **vector hóa văn bản** (tương tự Word2vec [16] nhưng cho cả tài liệu/đoạn văn thay vì chỉ từ).
    *   Hai kiến trúc chính:
        *   **PV-DM (Distributed Memory):** Tương tự CBOW trong Word2vec, dự đoán từ mục tiêu dựa trên các từ xung quanh và vector đại diện cho tài liệu.
        *   **PV-DBOW (Distributed Bag of Words):** Tương tự Skip-gram, dự đoán các từ trong ngữ cảnh dựa trên vector tài liệu (thường nhanh hơn và ít tốn bộ nhớ hơn).
    *   Trong bài báo này, Doc2vec được sử dụng để xây dựng vector đặc trưng cho mỗi đánh giá dựa trên mô hình trong thư viện **Gensim**. *Lưu ý: Cách thức chính xác Doc2vec kết hợp với "Opinion Words" không hoàn toàn rõ ràng từ mô tả và Hình 1. Có thể Doc2vec được áp dụng cho toàn bộ văn bản đã tiền xử lý, và "Opinion Words" được dùng để phân tích độ quan trọng sau này.*

*   **E. Phân loại Tình cảm (Sentiment Classification):**
    *   Sử dụng các thuật toán học máy tập hợp mạnh mẽ để phân loại các vector đánh giá thành "tích cực" hoặc "tiêu cực".
    *   Các thuật toán được sử dụng:
        *   **Cây Quyết định (Decision Trees - DTs) [2]:** Mô hình học có giám sát, tạo ra các quy tắc quyết định đơn giản.
        *   **Rừng Ngẫu nhiên (Random Forests - RF) [6]:** Phương pháp học tập hợp, kết hợp nhiều cây quyết định để cải thiện độ chính xác và chống overfitting.
        *   **Máy Tăng cường Gradient (Gradient Boosting Classifiers - GBC) [5, 10]:** Kỹ thuật học tập hợp, xây dựng các cây một cách tuần tự, mỗi cây mới cố gắng sửa lỗi của cây trước đó. Dựa trên ý tưởng của PAC Learning [17].

*   **Giải thích chi tiết:** Phần này mô tả chi tiết các bước thực hiện. Từ việc chuẩn bị dữ liệu, làm sạch, sử dụng công cụ để trích xuất từ quan điểm, biến đổi văn bản thành vector số bằng Doc2vec, và cuối cùng là áp dụng ba mô hình học máy tập hợp (DT, RF, GBC) để phân loại tình cảm.

---

**IV. Kết quả Thí nghiệm (Experimental Results)**

Phần này đánh giá hiệu quả của việc sử dụng tập từ ngữ quan điểm (thông qua các mô hình học máy) cho nhiệm vụ phân loại đánh giá.

*   **A. Phân loại Đánh giá với các Thuật toán Khác nhau:**
    *   **Hình 2:** So sánh độ chính xác (accuracy) trên tập huấn luyện (train_acc) và tập kiểm chứng (val_acc) của ba thuật toán (DTC, GBC, RF).
    *   **Quan sát:**
        *   **DTC:** Đạt độ chính xác 100% trên tập huấn luyện nhưng chỉ ~75.2% trên tập kiểm chứng -> **Overfitting nặng**.
        *   **GBC:** Độ chính xác huấn luyện thấp hơn (~99.3%) nhưng độ chính xác kiểm chứng cao nhất (~84.5%) -> **Tổng quát hóa tốt hơn**.
        *   **RF:** Độ chính xác huấn luyện cao (~99.3%), độ chính xác kiểm chứng khá tốt (~82.1%).
    *   **Kết luận:** GBC cho thấy hiệu suất tốt nhất trên dữ liệu chưa thấy, ít bị overfitting hơn so với DTC và RF trong trường hợp này.

**(Hình 2: So sánh độ chính xác của ba bộ phân loại DTC, RF, và GBC)**

*   **B. Các Từ ngữ Hữu ích để Xác định Ý nghĩa của Đánh giá:**
    *   Để xác định từ nào hữu ích nhất trong việc phân loại đánh giá là tích cực hay tiêu cực, các tác giả tính toán **điểm số quan trọng (important scores)** của các đặc trưng (từ ngữ) được trích xuất từ các bộ phân loại.
    *   **Hình 3, 4, 5:** Hiển thị 10 từ ngữ quan trọng nhất được tạo ra từ DTC, RF, và GBC.
    *   **Quan sát:**
        *   Các từ như **"good", "great", "better"** có độ quan trọng cao và xuất hiện trong top đầu của cả ba mô hình, cho thấy đây là những chỉ báo mạnh mẽ về tình cảm tích cực.
        *   Có sự khác biệt nhỏ ở các vị trí thấp hơn: GBC và RF có "happy", "love"; trong khi DTC có "healthy". Điều này cho thấy GBC và RF có thể nắm bắt các sắc thái tình cảm tích cực rộng hơn ("happy", "love") so với DTC ("healthy" có phạm vi hẹp hơn).
    *   So sánh giữa các mô hình cho thấy sự khác biệt trong cách chúng đánh giá tầm quan trọng của các từ ngữ.

**(Hình 3: Top 10 Đặc trưng Quan trọng từ Cây Quyết định)**
**(Hình 4: Top 10 Đặc trưng Quan trọng từ Rừng Ngẫu nhiên)**
**(Hình 5: Top 10 Đặc trưng Quan trọng từ Bộ phân loại Tăng cường Gradient)**

*   **Giải thích chi tiết:** Phần này trình bày kết quả thực nghiệm. Đầu tiên là so sánh hiệu năng của 3 mô hình học máy, cho thấy GBC hoạt động tốt nhất. Sau đó, phân tích sâu hơn bằng cách xem xét những từ ngữ nào được các mô hình xem là quan trọng nhất để đưa ra quyết định phân loại. Kết quả này xác nhận rằng các từ ngữ thể hiện tình cảm rõ ràng như "good", "great" là rất hữu ích.

---

**V. Kết luận (Conclusion)**

Bài báo đã giới thiệu một tập hợp các từ ngữ quan điểm và các từ ngữ hữu ích được trích xuất từ mô hình học máy để đánh giá xem một bài đánh giá sản phẩm thực phẩm trên Amazon là tiêu cực hay tích cực.

*   Phương pháp đề xuất có thể áp dụng cho các hệ thống thương mại điện tử khác.
*   Các đặc trưng được đề xuất (dựa trên từ ngữ quan điểm) đạt được kết quả hứa hẹn với các thuật toán học máy cổ điển.
*   Các từ ngữ hữu ích được trích xuất từ GBC và RF khá trực quan và phổ biến trong việc thể hiện cảm xúc về sản phẩm.

Với những thành tựu của kỹ thuật học sâu, nghiên cứu trong tương lai có thể tận dụng để đề xuất các mô hình phức tạp hơn nhằm cải thiện khả năng dự đoán.


---


