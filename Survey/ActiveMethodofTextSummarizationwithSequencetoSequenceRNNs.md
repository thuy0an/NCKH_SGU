# Active Method of Text Summarization with Sequence to Sequence RNNs

## Thông tin bài báo

\- Tác giả: Abu Kaisar M. Masum, Sheikh Abujar, Md. A.I. Talukder, AKM S.A. Rabby, Syed A. Hossain
\- Năm: 2019

\- Nguồn: Bangladesh, Daffodil International University
\- Mô hình: Bi-directional RNN with LSTM + Attention  

\- Dataset: Amazon Fine Food Reviews

\- Link: [Active Method of Text Summarization with Sequence to Sequence RNNs](https://www.researchgate.net/profile/Akm-Shahariar-Azad-Rabby/publication/338356977_Abstractive_method_of_text_summarization_with_sequence_to_sequence_RNNs/links/5e15a904a6fdcc283761cc0a/Abstractive-method-of-text-summarization-with-sequence-to-sequence-RNNs.pdf)
## Hiệu suất mô hình

\- Train loss = 0.036
## Kết quả chính

\- Giảm loss xuống còn 0.036, giúp mô hình tạo tóm tắt chính xác hơn.

\- Tóm tắt trôi chảy và dễ hiểu từ mô tả đánh giá thực phẩm trên Amazon.

\- Hoạt động tốt với văn bản ngắn, nhưng gặp khó khăn với văn bản dài.

\- Sử dụng mô hình Sequence-to-Sequence với Bi-directional RNN và Attention để cải thiện chất lượng tóm tắt.

\- Mô hình cho kết quả khả quan, nhưng vẫn có một số lỗi trong việc tạo tóm tắt.
## Hạn chế

\- Mô hình chỉ hoạt động tốt với các đoạn văn bản ngắn.

\- Không thể xử lý văn bản dài một cách hiệu quả.

\- Cần thời gian huấn luyện dài và phần cứng mạnh.


Tóm tắt (Abstract)

Bài báo này giải quyết vấn đề tóm tắt văn bản tự động, một lĩnh vực quan trọng trong Xử lý Ngôn ngữ Tự nhiên (NLP). Mục tiêu chính là tạo ra các bản tóm tắt trừu tượng - tức là bản tóm tắt được tạo ra bằng ngôn từ của mô hình, có thể chứa từ ngữ không có trong văn bản gốc, thay vì chỉ trích xuất câu - sao cho ngắn gọn, mạch lạc và dễ hiểu.

Nhóm tác giả đã sử dụng bộ dữ liệu "Amazon Fine Food Reviews" từ Kaggle. Họ áp dụng mô hình sequence-to-sequence (Seq2Seq), một kiến trúc phổ biến trong dịch máy và các tác vụ tạo văn bản khác. Cụ thể, mô hình bao gồm:
* Bộ mã hóa (Encoder): Sử dụng Mạng Nơ-ron Hồi quy (RNN) hai chiều (bi-directional) với các đơn vị LSTM (Long Short-Term Memory) để đọc và hiểu văn bản đầu vào (đánh giá sản phẩm).
* Bộ giải mã (Decoder): Sử dụng LSTM và cơ chế chú ý (attention mechanism) để tạo ra bản tóm tắt đầu ra. Cơ chế chú ý giúp bộ giải mã tập trung vào những phần quan trọng nhất của văn bản gốc khi tạo từng từ trong bản tóm tắt.

Bài báo cũng đề cập đến các thách thức như xử lý văn bản, quản lý từ vựng, xử lý từ không có trong từ điển, nhúng từ (word embedding), và tối ưu hiệu quả mô hình (giảm giá trị hàm mất mát - loss). Kết quả thực nghiệm cho thấy mô hình đã thành công giảm tổn thất huấn luyện xuống 0.036 và có khả năng tạo ra các bản tóm tắt tiếng Anh ngắn gọn.

---

Giới thiệu (Introduction)

Phần giới thiệu nhấn mạnh sự cần thiết của tóm tắt văn bản tự động để tiết kiệm thời gian và nâng cao hiệu quả so với việc con người đọc và tóm tắt thủ công. Bài báo phân biệt rõ hai phương pháp chính:
1. Tóm tắt Trích xuất (Extractive): Chọn lọc và kết hợp các câu hoặc cụm từ quan trọng từ văn bản gốc.
2. Tóm tắt Trừu tượng (Abstractive): Tạo ra bản tóm tắt mới bằng cách diễn giải nội dung văn bản gốc, có thể sử dụng từ ngữ khác. Phương pháp này khó hơn nhưng có khả năng tạo ra bản tóm tắt tự nhiên và mạch lạc hơn.

Bài báo này tập trung vào phương pháp tóm tắt trừu tượng. Nhóm tác giả đã điều chỉnh kiến trúc Seq2Seq với RNN hai chiều và cơ chế chú ý (cụ thể là Bahdanau attention), vốn thành công trong dịch máy, để áp dụng cho bài toán tóm tắt đánh giá thực phẩm.

---

Phương pháp luận (Methodology)

1. Dữ liệu: Sử dụng bộ dữ liệu Amazon Fine Food Reviews. Mặc dù bộ dữ liệu gốc rất lớn, nhóm tác giả chỉ sử dụng 20.000 mẫu cho thí nghiệm. Dữ liệu đầu vào là phần "Text" (nội dung đánh giá) và đầu ra mong muốn là phần "Summary" (bản tóm tắt do người dùng viết).
2. Tiền xử lý dữ liệu: Đây là bước quan trọng để làm sạch và chuẩn hóa dữ liệu văn bản:
    * Chuyển thành chữ thường.
    * Tách từ (Tokenization).
    * Mở rộng các dạng viết tắt (ví dụ: "don't" -> "do not").
    * Loại bỏ nhiễu bằng biểu thức chính quy.
    * Loại bỏ các từ dừng (stop words) phổ biến nhưng ít mang nghĩa (như "the", "a", "is").
    * Chuẩn hóa từ (Lemmatization): Đưa các từ về dạng gốc (ví dụ: "running", "ran" -> "run").
3. Từ vựng và Nhúng từ (Vocabulary & Word Embedding):
    * Xây dựng từ vựng từ dữ liệu đã tiền xử lý.
    * Lọc bỏ các từ có tần suất xuất hiện thấp (chỉ giữ lại từ xuất hiện > 20 lần).
    * Sử dụng nhúng từ được huấn luyện trước (pre-trained word embeddings) - cụ thể là ConceptNet Numberbatch - để biểu diễn mỗi từ bằng một vector số, giúp mô hình hiểu được mối quan hệ ngữ nghĩa giữa các từ.
4. Kiến trúc Mô hình:
    * Encoder-Decoder: Kiến trúc Seq2Seq tổng thể.
    * Encoder: RNN/LSTM hai chiều đọc chuỗi đầu vào và tạo ra các trạng thái ẩn (hidden states) chứa thông tin ngữ cảnh.
    * Decoder: RNN/LSTM tạo chuỗi đầu ra từng từ một.
    * Attention Mechanism (Bahdanau): Cho phép bộ giải mã xem xét tất cả các trạng thái ẩn của bộ mã hóa và tập trung vào những trạng thái ẩn liên quan nhất tại mỗi bước tạo từ.
    * Special Tokens: Sử dụng các token đặc biệt như <PAD> (đệm), <EOS> (kết thúc chuỗi), <GO> (bắt đầu chuỗi giải mã), <UNK> (từ không biết).

---

Thí nghiệm và Kết quả (Experiments and Results)

* Thiết lập: Sử dụng Tensorflow 1.12.0. Các siêu tham số chính bao gồm: 100 epochs, batch size 64, kích thước ẩn RNN 256, tốc độ học 0.005, dropout 0.25, bộ tối ưu hóa Adam.
* Phân chia dữ liệu: 16.000 mẫu (80%) cho huấn luyện, 4.000 mẫu (20%) cho kiểm tra.
* Kết quả:
    * Mô hình đạt được tổn thất huấn luyện thấp (0.036).
    * Mô hình tạo ra các bản tóm tắt nhìn chung là tốt, mạch lạc và "tích cực", dù vẫn còn một số lỗi.
    * Các ví dụ cụ thể (Bảng 2, 3 trong bài báo) cho thấy khả năng tóm tắt trừu tượng của mô hình (ví dụ: tạo ra cụm "not for best" hoặc "tasted plain" dựa trên nội dung gốc).

---

Kết luận (Conclusion)

Bài báo đã trình bày thành công một phương pháp tóm tắt văn bản trừu tượng sử dụng mô hình Seq2Seq với RNN/LSTM hai chiều và cơ chế chú ý. Mô hình có khả năng tạo ra các bản tóm tắt ngắn gọn, mạch lạc cho các bài đánh giá sản phẩm bằng tiếng Anh.

Tuy nhiên, các hạn chế bao gồm:
* Khả năng xử lý văn bản dài còn hạn chế.
* Yêu cầu độ dài đầu vào và đầu ra cố định (do padding).
* Đôi khi tạo ra tóm tắt chưa hoàn toàn chính xác.
* Cần tài nguyên tính toán và thời gian huấn luyện đáng kể.

---

Công việc Tương lai (Future Work)

Nhóm tác giả đề xuất các hướng phát triển trong tương lai:
* Cải thiện mô hình để xử lý văn bản có độ dài thay đổi và văn bản dài hơn.
* Mục tiêu quan trọng là áp dụng và phát triển kỹ thuật này cho ngôn ngữ Bengali. Điều này bao gồm việc giải quyết thách thức thiếu hụt tài nguyên NLP cho tiếng Bengali (như word embeddings, công cụ lemmatization) và đóng góp vào việc xây dựng các tài nguyên này.

