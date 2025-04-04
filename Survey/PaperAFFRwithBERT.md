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


**Tóm tắt (Abstract)**

Đánh giá trực tuyến đóng vai trò thiết yếu cho cả người tiêu dùng và doanh nghiệp. Thông qua việc đọc đánh giá, mọi người có thể tìm hiểu về sản phẩm một cách trực quan. Ngày càng có nhiều người dựa vào đánh giá để hiểu về sản phẩm và dịch vụ trước khi mua hàng, đặc biệt là trong ngành thương mại điện tử. Thông thường, đánh giá gồm hai phần: điểm số tổng thể và mô tả bằng văn bản.

Trong bài báo nghiên cứu này, các tác giả phát triển một mô hình dựa trên **BERT (Bidirectional Encoder Representations from Transformers)**, một trong những phương pháp Xử lý Ngôn ngữ Tự nhiên (NLP) tiên tiến, để **dự đoán điểm đánh giá tổng thể (overall review score)** dựa trên phần mô tả bằng văn bản. Bộ dữ liệu được áp dụng là "Amazon Fine Food Reviews" trên Kaggle.

Trước khi phân tích mô hình, nhóm tác giả đã **làm sạch (cleansed)** bộ dữ liệu, loại bỏ dấu câu và ký tự đặc biệt trong bình luận, chọn lại dữ liệu dựa trên độ dài của đánh giá và tạo **đám mây từ (word clouds)**. Mô hình đạt được **độ chính xác (accuracy) 0.7982** và **giá trị mất mát (loss value) 0.5433** khi dự đoán trên nhóm dữ liệu kiểm tra (testing group). Mô hình này có thể được ứng dụng trong các doanh nghiệp thực phẩm để tạo ra điểm số tổng thể từ các đánh giá văn bản thu thập từ nhiều nguồn khác nhau.

*   **Giải thích chi tiết:** Phần tóm tắt nêu bật tầm quan trọng của đánh giá trực tuyến. Mục tiêu chính của bài báo là sử dụng mô hình NLP mạnh mẽ (BERT) để tự động dự đoán điểm số (ví dụ: từ 1 đến 5 sao) mà khách hàng sẽ cho dựa trên nội dung văn bản họ viết. Họ đã thực hiện các bước làm sạch dữ liệu cẩn thận và đạt được độ chính xác khá tốt (gần 80%) trên bộ dữ liệu đánh giá thực phẩm của Amazon.

---

**Từ khóa (Keywords):** Đánh giá thực phẩm Amazon; BERT; làm sạch dữ liệu; đám mây từ; Tinh chỉnh (Fine-Tuning); NLP.

---

**1. Giới thiệu (Introduction)**

Sự phát triển nhanh chóng của Internet đã thay đổi cách mọi người tiếp cận thông tin. Thay vì đọc tạp chí hay báo giấy, mọi người ngày càng sử dụng Internet để tìm kiếm thông tin, đọc tin tức, khám phá địa điểm, mua sắm và tìm hiểu về những điều mới lạ. Đặc biệt, việc xem và đăng đánh giá về nhà hàng, thực phẩm trên các trang web trở nên rất phổ biến.

Theo các báo cáo (Financesonline, Pew Research Center), phần lớn người tiêu dùng (97%) đọc đánh giá trực tuyến và nhiều người (89%) dành thời gian xem xét đánh giá trước khi mua hàng. Đánh giá giúp người tiêu dùng đưa ra quyết định mua hàng. Một bài đánh giá thường bao gồm văn bản mô tả cảm nhận của người viết và một điểm số (thường từ 1 đến 5, 1 là tệ nhất, 5 là tốt nhất).

Nghiên cứu này tập trung vào việc xây dựng một mô hình để **dự đoán điểm số đánh giá dựa trên nội dung văn bản**. Nhóm tác giả sử dụng bộ dữ liệu "Amazon Fine Food Review" và mô hình NLP nổi tiếng là **BERT**. Bài báo được cấu trúc gồm: Mô tả dữ liệu và quá trình xử lý (Phần 2), Phân tích mô hình (Phần 3), và Kết luận (Phần 4).

*   **Giải thích chi tiết:** Phần giới thiệu nhấn mạnh xu hướng sử dụng đánh giá trực tuyến ngày càng tăng. Nó xác định rõ mục tiêu của nghiên cứu: không chỉ phân loại đánh giá là tích cực/tiêu cực (như bài báo trước), mà là dự đoán chính xác *điểm số cụ thể* (1-5) mà người dùng đã cho, chỉ dựa vào phần bình luận bằng chữ của họ. Mô hình được chọn là BERT, một công cụ mạnh mẽ trong NLP.

---

**2. Dữ liệu (Data)**

*   **2.1. Mô tả dữ liệu (Data description):**
    *   Bộ dữ liệu gốc gồm 568.454 đánh giá thực phẩm trên Amazon, với 10 cột dữ liệu như "ProductId", "Score", "Time", "Summary", "Text". Thời gian dữ liệu kéo dài hơn 10 năm (1999-2012).
    *   **Bảng 1:** Mô tả thống kê các biến định lượng chính ("Id", "HelpfulnessNumerator", "HelpfulnessDenominator", "Score", "Time"). Đáng chú ý là biến "Score" có giá trị nhỏ nhất là 1, lớn nhất là 5, nhưng phân vị thứ nhất (25%) là 4, trung vị (50%) và phân vị thứ ba (75%) đều là 5, cho thấy dữ liệu bị **lệch nặng về điểm số cao (đặc biệt là 5)**.

**(Bảng 1: Mô tả bộ dữ liệu)**

*   **2.2. Làm sạch dữ liệu (Data cleaning):**
    *   **2.2.1.1. Giá trị bị thiếu (Missing values):** Kiểm tra và xử lý các giá trị thiếu. Phát hiện thiếu dữ liệu ở cột "ProfileName" và "Summary". Do "ProfileName" không cần thiết, chỉ xử lý cột "Summary" (có 27 giá trị thiếu) bằng cách thay thế bằng khoảng trắng. Sau đó, tạo một cột mới tên "Combo" bằng cách **kết hợp nội dung từ cột "Summary" và "Text"**.
    *   **2.2.1.2. Ngôn ngữ và giá trị trùng lặp (Language and duplicated values):** Chỉ giữ lại các bình luận bằng tiếng Anh. Loại bỏ các bản ghi trùng lặp dựa trên cột "Combo". Bộ dữ liệu mới còn 394.970 đánh giá.
    *   **2.2.1.3. Dấu câu và ký tự đặc biệt (Punctuation and special character):** Loại bỏ các cột không cần thiết, chỉ giữ lại "Combo" và "Score". Làm sạch cột "Combo" bằng cách loại bỏ tất cả dấu câu và ký tự đặc biệt (?, !, #...).
    *   **2.2.2. Thời gian (Time):** Chuyển đổi định dạng thời gian từ Unix Timestamp sang "năm-tháng-ngày". Tạo biến "Year". Chỉ giữ lại dữ liệu từ năm 2010 trở đi để đảm bảo tính cập nhật và phù hợp với phân tích.

*   **2.3. Phân phối điểm số (Score distribution):**
    *   Chuyển đổi điểm số thành số nguyên (nếu cần).
    *   **Bảng 2:** Cho thấy sự phân phối của các điểm số từ 1 đến 5. Điểm 5 chiếm tỷ lệ áp đảo (62.6%), trong khi các điểm 1, 2, 3, 4 chiếm tỷ lệ lần lượt là 9.9%, 5.5%, 7.8%, và 14.2%. Điều này xác nhận **sự mất cân bằng nghiêm trọng** của dữ liệu.

**(Bảng 2: Thống kê điểm số)**

*   **2.3.1. Chọn lại dữ liệu dựa trên độ dài đánh giá (Data reselection based on length of reviews):**
    *   Độ dài văn bản đánh giá dao động từ 43 đến 21.409 ký tự.
    *   Để tránh các đánh giá quá chung chung (ví dụ: "OMG! This food is so good!"), chỉ giữ lại những đánh giá có **độ dài hơn 100 ký tự**. Mục đích là để mô hình học từ những bình luận chi tiết hơn.

*   **2.3.2. Lấy mẫu lại (Resampling):**
    *   Sự mất cân bằng điểm số (76.8% là điểm 4 hoặc 5) có thể làm mô hình bị thiên vị, dễ dàng dự đoán điểm 5 để đạt độ chính xác cao một cách giả tạo.
    *   Để khắc phục, áp dụng phương pháp **lấy mẫu lại (resampling)**: chọn ngẫu nhiên **15.000 bản ghi cho mỗi mức điểm (1, 2, 3, 4, 5)**.
    *   Bộ dữ liệu cuối cùng dùng để huấn luyện và đánh giá mô hình gồm **75.000 bản ghi**, cân bằng về số lượng cho mỗi loại điểm số.

**(Hình 1 (trang 4): Biểu đồ phân phối điểm số ban đầu, cho thấy sự mất cân bằng)**

*   **Giải thích chi tiết:** Phần này mô tả rất kỹ lưỡng quá trình chuẩn bị dữ liệu, bao gồm xử lý giá trị thiếu, loại bỏ dữ liệu không phù hợp (ngôn ngữ khác, trùng lặp, dấu câu, đánh giá cũ, đánh giá quá ngắn), và đặc biệt là giải quyết vấn đề mất cân bằng dữ liệu nghiêm trọng bằng cách lấy mẫu lại để tạo ra một bộ dữ liệu cân bằng hơn cho việc huấn luyện mô hình. Đây là những bước cực kỳ quan trọng để đảm bảo chất lượng và độ tin cậy của mô hình học máy.

---

**3. Phân tích Mô hình (Model analysis)**

*   **3.1. Phân chia tập Huấn luyện, Xác thực & Kiểm tra (Training, validation & testing groups):**
    *   Chia bộ dữ liệu đã xử lý (75.000 bản ghi) thành 3 tập theo tỷ lệ 7:1:2:
        *   Tập huấn luyện (Training): 70% (52.500 bản ghi) - Dùng để huấn luyện mô hình.
        *   Tập xác thực (Validation): 10% (7.500 bản ghi) - Dùng để tinh chỉnh siêu tham số và theo dõi quá trình huấn luyện.
        *   Tập kiểm tra (Testing): 20% (15.000 bản ghi) - Dùng để đánh giá hiệu năng cuối cùng của mô hình trên dữ liệu hoàn toàn mới.

*   **3.2. Đám mây từ (Words cloud):**
    *   **Hình 2:** Hiển thị đám mây từ được tạo ra từ nội dung đánh giá trước khi huấn luyện BERT. Các từ có kích thước lớn hơn xuất hiện thường xuyên hơn.
    *   Các từ khóa nổi bật: "well", "gluten-free", "taste like", "great product", "know", "found", "good", "delicious".
    *   Điều này cho thấy người dùng thường quan tâm đến các vấn đề như sản phẩm không chứa gluten, hương vị, chất lượng sản phẩm, và so sánh giữa các loại thực phẩm.

**(Hình 2: Đám mây từ)**

*   **3.3. Mô hình BERT (BERT model):**
    *   Giới thiệu BERT [8] là một mô hình **tiền huấn luyện (pre-training)** không giám sát cho NLP, dựa trên kiến trúc Transformer và có khả năng hiểu ngữ cảnh hai chiều (bidirectional).
    *   **Hình 3:** Minh họa kiến trúc của BERT.
    *   BERT có hai giai đoạn:
        1.  **Tiền huấn luyện:** Huấn luyện trên một lượng lớn dữ liệu văn bản thô với các nhiệm vụ như Masked Language Modeling (dự đoán từ bị che) và Next Sentence Prediction (dự đoán câu tiếp theo) [9]. Giai đoạn này giúp mô hình học được các biểu diễn ngôn ngữ sâu sắc.
        2.  **Tinh chỉnh (Fine-tuning):** Điều chỉnh mô hình đã được tiền huấn luyện cho một nhiệm vụ cụ thể (như phân loại văn bản, hỏi đáp...).
    *   Trong nghiên cứu này, các tác giả sử dụng **tính năng tinh chỉnh của BERT cho nhiệm vụ phân loại (classification)** - tức là dự đoán điểm số từ 1 đến 5.

**(Hình 3: Kiến trúc của BERT)**

*   **Quá trình Tinh chỉnh và Kết quả (Fine-tuning Process and Results - Trang 5):**
    *   Chọn mô hình tiền huấn luyện: **"bert-base-cased"**.
    *   **Tokenization:** Tách văn bản đánh giá thành các token mà BERT có thể hiểu.
    *   **One-Hot Encoding:** Mã hóa điểm số (1-5) thành dạng vector nhị phân để làm đầu ra cho mô hình.
    *   Thiết lập siêu tham số: Tốc độ học (learning rate) rất nhỏ (1e-5), chỉ thực hiện **3 epochs** (lượt huấn luyện). Lý do dùng ít epochs là vì mô hình BERT đã được tiền huấn luyện rất tốt, chỉ cần tinh chỉnh nhẹ nhàng cho nhiệm vụ mới.
    *   **Kết quả trên các tập dữ liệu:**
        *   Tập huấn luyện: Loss = 0.477, Accuracy = 0.816
        *   Tập xác thực: Loss = 0.549, Accuracy = 0.798
        *   **Tập kiểm tra (Kết quả cuối cùng): Loss = 0.5433, Accuracy = 0.7982**
    *   **Hình 4:** Hiển thị các biểu đồ về loss và accuracy trong quá trình huấn luyện và xác thực, cho thấy mô hình hội tụ tốt.

**(Hình 4: Biểu đồ hiệu năng của mô hình BERT)**

*   **Giải thích chi tiết:** Phần này giải thích cách mô hình BERT được áp dụng. Đầu tiên, dữ liệu được chia hợp lý. Đám mây từ cung cấp cái nhìn trực quan về nội dung đánh giá. Sau đó, giải thích về BERT và cách nó được tinh chỉnh cho nhiệm vụ dự đoán điểm số. Kết quả cuối cùng trên tập kiểm tra cho thấy độ chính xác gần 80%, một con số đáng kể đối với một nhiệm vụ phức tạp và chủ quan như dự đoán điểm đánh giá từ văn bản.

---

**4. Kết luận và Thảo luận thêm (Conclusion and further discussion)**

*   **4.1. Kết luận (Conclusion):**
    *   Mô hình BERT đạt độ chính xác 0.7982 và loss 0.5433 trên tập kiểm tra khi dự đoán điểm số từ văn bản đánh giá.
    *   Kết quả này rất ấn tượng vì các đánh giá mang tính chủ quan cao và được viết bởi nhiều người khác nhau (cùng một sản phẩm/dịch vụ có thể nhận điểm khác nhau).
    *   Mô hình BERT rất cần thiết cho các doanh nghiệp ngành thực phẩm để đánh giá hiệu suất và phát triển sản phẩm. Nó có tính thực tiễn cao, giúp các công ty ước tính hiệu suất tổng thể tại một địa điểm cụ thể hoặc trên toàn quốc từ các nguồn đánh giá khác nhau (Yelp, Amazon, khảo sát...). Công ty có thể trích xuất các từ/cụm từ thường gặp từ các đánh giá điểm thấp/cao để điều chỉnh hương vị/thành phần cho phù hợp với thị hiếu địa phương.

*   **4.2. Thảo luận thêm (Further discussion):**
    *   **Hạn chế về dữ liệu:** Nghiên cứu chỉ sử dụng dữ liệu từ Amazon. Việc thu thập thêm dữ liệu từ các nguồn khác (Yelp, Google Forms, khảo sát...) có thể giúp xây dựng mô hình mạnh mẽ và chính xác hơn.
    *   **Vấn đề đánh giá giả mạo (Fake reviews):** Đánh giá giả có thể ảnh hưởng đến độ chính xác của mô hình. Việc phát triển và tích hợp một mô hình phân loại đánh giá giả để lọc dữ liệu đầu vào có thể cải thiện hiệu suất của mô hình BERT.
    *   **Hạn chế về mô hình:** Chỉ sử dụng "bert-base-cased". Có thể khám phá các mô hình Transformer tiên tiến khác (như RoBERTa [11], vốn được huấn luyện trên nhiều dữ liệu hơn BERT) cho nhiệm vụ phân loại văn bản để cải thiện hơn nữa.

