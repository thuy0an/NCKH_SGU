
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
| Naive Bayes          | 83.43 | 82.35 | 88.78 | 85.54 |
| LinearSVC         | 88.38 | 88.54 | 88.39 | 88.46 |

## Kết quả chính
- Sử dụng Apache Spark để xử lý dữ liệu lớn trong phân tích cảm xúc.
- Linear SVC có độ chính xác cao nhất (> 88%).
- Hệ thống có thể phân loại đánh giá thành **tích cực** hoặc **tiêu cực** với hiệu suất tối ưu.

## Hạn chế
- Chưa thử nghiệm với các mô hình học sâu như BERT.
- Cần cải thiện hiệu suất trên tập dữ liệu lớn hơn.

Okay, đây là bản dịch và phân tích chi tiết bài báo khoa học "Sentiment Analysis of Online Food Reviews using Big Data Analytics".

---



**Tóm tắt:** Ngày nay, phân tích tình cảm đã trở nên rất quan trọng, chủ yếu được sử dụng cho các bộ dữ liệu khổng lồ và hữu ích cho các nhà nghiên cứu trong việc áp dụng các phương pháp và kỹ thuật. Dữ liệu thực phẩm của Amazon đang tăng trưởng theo cấp số nhân và các hệ thống truyền thống không thể xử lý nó, vì vậy chúng tôi đã sử dụng Dữ liệu Lớn (Big Data) để khắc phục vấn đề này. Trong bài báo này, chúng tôi khám phá các phương pháp và kỹ thuật phân tích tình cảm khác nhau bằng cách sử dụng hệ thống xử lý dữ liệu Apache Spark cho các bộ dữ liệu lớn về đánh giá Fine Food của Amazon. Ba cơ chế được áp dụng có độ chính xác hơn 80% là Linear SVC, Logistic Regression và Naïve Bayes bằng cách sử dụng MLlib, thư viện ML của Apache Spark. Khi áp dụng các phương pháp này, chúng tôi nhận thấy rằng Linear SVC hoạt động hiệu quả hơn NB và Logistic Regression.

**Từ khóa:** Phân tích tình cảm; Apache Spark; đánh giá, Học máy, Dữ liệu lớn, Phân tích

**I. GIỚI THIỆU**

Đánh giá trực tuyến cho doanh nghiệp của bạn là một trong những yếu tố quan trọng nhất trong phân tích tiếp thị. Trong kinh doanh, đánh giá của khách hàng trực tuyến trở nên rất quan trọng đối với sản phẩm và dịch vụ, do đó, chúng ta có thể theo dõi các đánh giá xấu và tốt, qua đó phân tích chất lượng sản phẩm và tiêu chuẩn của chúng, cũng như hữu ích trong việc tạo ra các phương pháp và kỹ thuật mới để cải thiện chất lượng sản phẩm [1]. Đánh giá của khách hàng là về phản hồi vì nó chứa một lượng lớn dữ liệu đang lan truyền rộng rãi mỗi giây và chứa dữ liệu có cấu trúc, phi cấu trúc và tình cảm, được gọi là phân tích dữ liệu lớn và trích xuất thông tin. Sự thành công của một công ty hoặc sản phẩm phụ thuộc trực tiếp vào phản hồi của khách hàng [2].

Phân tích tình cảm trong lĩnh vực truy xuất thông tin là việc xác định và phân loại các ý kiến từ một đoạn văn bản dưới dạng tích cực hoặc tiêu cực một cách tính toán. Đối với lượng dữ liệu khổng lồ, việc phân tích và sử dụng đủ dữ liệu cho chức năng của chúng trở nên rất quan trọng và do đó giờ đây nó là một nhiệm vụ rất lớn đối với kho dữ liệu và cơ sở dữ liệu quan hệ [3]. Phân tích dữ liệu lớn có ba khía cạnh quan trọng là **Volume (Khối lượng)**, **Velocity (Tốc độ)** và **Variety (Đa dạng)**. Thu thập kích thước dữ liệu khổng lồ từ các nội dung dữ liệu khác nhau tại một thời điểm cụ thể được gọi là **Volume**. **Velocity** là tốc độ mà dữ liệu có thể được đo lường [4]. Các loại dữ liệu khác nhau có nguồn khác nhau bao gồm dữ liệu có cấu trúc và phi cấu trúc như văn bản, âm thanh, video và hình ảnh được công nhận là **Variety** của dữ liệu [5]. Hầu hết các robot, các cơ chế và kỹ thuật học máy phức tạp được sử dụng cho các yêu cầu của dữ liệu lớn. Vì lượng dữ liệu rất lớn, không thể sử dụng trong bộ nhớ của máy tính cá nhân, do đó một số công cụ học máy như R và Weka có thể được sử dụng để phân tích dữ liệu lớn. Một số công cụ mới hiện đã được giới thiệu như Apache Spark, Apache Hadoop, những công cụ này có thể dễ dàng xử lý các thuật toán học máy và hoạt động rất hiệu quả và có thể đạt được hiệu suất tốc độ cao [6].

Apache Spark được giới thiệu bởi Đại học California vào năm 2009. Nó có thể được sử dụng chủ yếu cho dữ liệu kích thước lớn, nhưng nó có thể hoạt động hiệu quả cho cả dữ liệu batch và streaming, cũng như dễ dàng xử lý API trên các bộ dữ liệu khổng lồ [7]. Spark là một framework hiệu quả nhất cho dữ liệu lớn so với các tối ưu hóa khác như Hadoop và đạt được hiệu suất cao. Spark MLlib là một thư viện có khả năng mở rộng và cũng có thể được sử dụng với các ngôn ngữ lập trình cấp cao khác nhau [8]. Ứng dụng của Spark bao gồm 5 thực thể cơ bản được thể hiện trong Hình 01. (Hình 01 mô tả kiến trúc Spark cơ bản với Driver Program, Cluster Manager và Worker Nodes).

Việc tạo ra tin tức giả, bình luận và tình cảm đang gia tăng hàng ngày từ các trang web và phương tiện truyền thông xã hội [9]. Phân tích tình cảm được sử dụng để giải quyết các vấn đề học máy khác nhau [10]. Trong bài báo này, chúng tôi đã sử dụng Spark MLlib vì đây là một thư viện hiện đại được tạo ra vào năm 2014. Nghiên cứu đã được thực hiện bằng cách sử dụng MLlib của Spark và nó có thể hữu ích cho phân tích dữ liệu lớn.
Nghiên cứu rất hạn chế đã được thực hiện trên Spark MLlib cho dữ liệu kích thước lớn nhưng cần nhiều công việc hơn trong lĩnh vực này. Bằng cách sử dụng các kỹ thuật và phương pháp khác nhau trên MLlib của Spark, khám phá những cách khác nhau mới để phân tích dữ liệu lớn nhằm đạt được hiệu quả và hiệu suất cao.
Theo hiểu biết của chúng tôi, công trình của chúng tôi là công trình đầu tiên về đánh giá tình cảm liên quan đến thực phẩm Amazon trên dữ liệu lớn bằng cách sử dụng học máy.

**II. CÔNG TRÌNH LIÊN QUAN**

Các nhà khoa học đang tích cực nghiên cứu phân tích tình cảm, lĩnh vực đã trở thành lĩnh vực nghiên cứu lớn nhất trong vài năm qua. Sultana, Kumar [11] mô tả rằng phân tích tình cảm có ba khía cạnh quan trọng: tích cực, tiêu cực và trung tính. Từ vài năm qua, web thế giới trở thành yếu tố chính của đánh giá khách hàng, thông qua mạng xã hội và các trang web thương mại điện tử, như Facebook, người dùng Twitter có thể chia sẻ đánh giá của họ và những đánh giá này có thể tốt hoặc xấu, và những đánh giá này giúp đưa ra lựa chọn về việc áp dụng kế hoạch mới và quyết định về sản phẩm. Chen, Xue [12] giới thiệu một kỹ thuật mới để loại bỏ các đặc điểm của phân tích tình cảm đối với đánh giá sản phẩm... (Tóm tắt các công trình liên quan khác [13-21], nhấn mạnh các kỹ thuật như TF-IDF, heuristic methods, Naïve Bayes, SVM, Logistic Regression, và việc áp dụng chúng vào các bộ dữ liệu khác nhau như Twitter, đánh giá phim, đánh giá nhà hàng, dự đoán thị trường chứng khoán, v.v., và một số nghiên cứu cũng sử dụng Spark).

**III. BỘ DỮ LIỆU VÀ PHƯƠNG PHÁP LUẬN**

Phương pháp tiếp cận được sử dụng trong bài báo này có năm giai đoạn được thể hiện trong Hình 02. Các giai đoạn này là thu thập bộ dữ liệu thông qua trực quan hóa khác nhau, tiền xử lý dữ liệu, trích xuất đặc trưng, triển khai các bộ phân loại học máy thông qua Spark MLlib và cuối cùng, đánh giá các mô hình thông qua chia tách tập huấn luyện-kiểm tra (train-test split) bằng cách sử dụng các số liệu khác nhau của phân loại nhị phân.

*(Hình 02: Sơ đồ quy trình: Data Collection -> Pre-processing (Data Cleaning, Normalization, Tokenization, Stop words removal, Stemming) -> Feature Vectors -> Spark MLlib Classifiers -> Result Evaluation. Có nhánh Data Exploration & Visualization từ Data Collection)*

**A. Bộ dữ liệu (Dataset)**
Bộ dữ liệu Fine Food của Amazon [22] được sử dụng cho các thí nghiệm. Bộ dữ liệu của Amazon bao gồm 568.454 đánh giá, số lượng người dùng là 256.059, số sản phẩm là 74.258 và số cột là 10. Các đặc trưng bao gồm: product_id duy nhất, user_id duy nhất, tên hồ sơ (profile name), số người dùng thấy đánh giá hữu ích (helpfulness numerator), số người dùng cho biết đánh giá có hữu ích hay không (helpfulness denominator), điểm số dựa trên xếp hạng từ 1 đến 5, dấu thời gian (timestamp) của đánh giá, tóm tắt đánh giá và văn bản đánh giá.

**B. Giai đoạn tiền xử lý (Preprocessing Stage)**
Trước khi huấn luyện các mô hình, bộ dữ liệu được đưa vào giai đoạn tiền xử lý để cung cấp đầu vào tốt nhất cho các mô hình huấn luyện. Có các bước làm sạch dữ liệu cần thiết sau đây, còn được gọi là giai đoạn xử lý dữ liệu thô (data wrangling).
*   Thứ nhất, xác định và loại bỏ tất cả các giá trị null và trùng lặp khỏi văn bản.
*   Thứ hai, loại bỏ nhiễu khỏi dữ liệu văn bản bằng cách loại bỏ dữ liệu không liên quan có thể làm giảm hiệu suất của các bộ phân loại như ký tự không phải chữ cái, chữ số, ký tự đặc biệt và dấu câu.
*   Thứ ba, chuẩn hóa cột điểm số có giá trị từ 1 đến 5 (Hình 03). *(Hình 03: Biểu đồ phân phối điểm số, cho thấy điểm 5 chiếm đa số)*. Để làm điều này, tạo một cột mới tên là 'label' có giá trị 0 hoặc 1 dựa trên các đặc trưng hữu ích (helpfulness features) như trong Hình 04. *(Hình 04: Biểu đồ thể hiện mức độ hữu ích so với điểm số, cho thấy sự không cân bằng)*
*   Cuối cùng, **tokenize** (tách từ) văn bản đã tiền xử lý dựa trên khoảng trắng và loại bỏ các **stop words** (từ dừng) là những từ phổ biến và có ít ý nghĩa trong câu như giới từ và liên từ (Bảng 01). *(Bảng 01: Ví dụ về văn bản trước và sau khi tokenize và loại bỏ stop words)*

**C. Khám phá dữ liệu (Data Exploration)**
Khám phá và trực quan hóa dữ liệu là một giai đoạn quan trọng cho phân tích. Chúng tôi đã sử dụng thư viện matplotlib và seaborn để khám phá dữ liệu.
Hình 05 hiển thị ma trận tương quan giữa năm đặc trưng số. *(Hình 05: Ma trận tương quan)*
Đặc trưng tóm tắt (summary) rất quan trọng chứa nội dung đánh giá, Hình 06 cho thấy các thuật ngữ thường xuyên nhất là tích cực như love, good, best, taste, delicious và yummy. *(Hình 06: Đám mây từ cho các thuật ngữ phổ biến nhất trong các đánh giá)*
Hình 07 cho thấy các thuật ngữ thường xuyên đối với điểm số thấp (là 1). Các thuật ngữ như garbage, price, horrible, bad, nasty, rip, tasteless và warning là các tình cảm tiêu cực với điểm số 1. *(Hình 07: Đám mây từ cho các thuật ngữ phổ biến với điểm số 1)*
Hình 08 cho thấy các thuật ngữ thường xuyên đối với điểm số cao (là 5). Các thuật ngữ great, good, oatmeal, spice, tasty, delicious, và food là các tình cảm tích cực với điểm số 5. *(Hình 08: Đám mây từ cho các thuật ngữ phổ biến với điểm số 5)*

**D. Trích xuất đặc trưng (Feature extraction)**
Kỹ thuật đặc trưng (Feature engineering) là một giai đoạn quan trọng khác trước khi huấn luyện mô hình. Dựa trên phần khám phá dữ liệu, Hình 04 cho thấy các đánh giá bị lệch về phía tích cực, một nửa số đánh giá có số phiếu bầu bằng không và nhiều người đồng ý với điểm số cao nhất. Vì vậy, chúng tôi đã chuyển đổi điểm số thành hai lớp, nhãn tích cực và tiêu cực như trong Hình 09. *(Hình 09: Biểu đồ cột số lượng lớp tích cực và tiêu cực sau khi chuẩn hóa và cân bằng)*
Dữ liệu văn bản được chuyển đổi thành vector đặc trưng trước khi huấn luyện mô hình để các mô hình có thể được huấn luyện hiệu quả. **TF-IDF** (Term Frequency-Inverse Document Frequency) được sử dụng trong trích xuất đặc trưng. TF đếm số lần từ xuất hiện trong văn bản và IDF đếm mức độ phổ biến và hiếm của từ trong tất cả các tài liệu hoặc văn bản.

**E. Mô hình phân loại SparkML (SparksML Classifiers models)**
Các bộ phân loại khác nhau được huấn luyện và đánh giá nhưng 3 bộ phân loại có độ chính xác hơn 80% được chọn trong bài báo này: Logistic Regression (LR), Linear Support Vector Classifier (LinearSVC), và Naïve Bayes (NB).
*   **LinearSVC:** Là phương pháp học máy phù hợp nhất. Đây chủ yếu là kỹ thuật phân loại các vấn đề tuyến tính. Mục tiêu của LinearSVC là tìm siêu phẳng (hyperplane) tốt nhất phân loại dữ liệu. LinearSVC có khả năng thực thi thích ứng SVC với kernel tuyến tính. Trong sklearn, quy trình được sử dụng trong LinearSVC chính xác hơn [23].
*   **Naïve Bayes (NB):** Dựa trên định lý Bayes, là kỹ thuật học máy có giám sát được sử dụng cho các tác vụ phân loại. Định lý Bayes được giải thích như sau: P(a|b) = [P(b|a) * P(a)] / P(b) (Công thức 1). "a" và "b" là các sự kiện độc lập... Naïve Bayes xây dựng mô hình bằng cách cố định phân phối của mọi đặc trưng [24, 25].
*   **Logistic Regression (LR):** Là một thuật toán học máy được các nhà nghiên cứu sử dụng rộng rãi. Hồi quy logistic hiểu vector biến và tìm ra hệ số cho các biểu thức đầu vào và sau đó theo dõi lớp của văn bản dưới dạng vector từ. Hàm hồi quy logistic xác định nhiều hàm tuyến tính được biểu thị dưới dạng Logit(P) = β0 + β1X1 + β2X2 + ... + βkXk (Công thức 2). P đại diện cho xác suất xảy ra của đặc trưng... [19, 26].

**IV. THIẾT LẬP THÍ NGHIỆM**

Chúng tôi đã sử dụng Databricks Spark cloud với Python 3.0. Có 82007 mẫu tiêu cực và 486447 mẫu tích cực trong các lớp của chúng tôi. Để cân bằng các lớp từ tổng bộ dữ liệu, 82007 mẫu cho lớp tích cực và cùng kích thước mẫu cho lớp tiêu cực được lấy cho các thí nghiệm. Bằng cách chia ngẫu nhiên 80% để huấn luyện và 20% để kiểm tra từ tổng số. Hình 10 cho thấy tóm tắt bộ dữ liệu theo nhãn lớp tích cực và tiêu cực. *(Hình 10: Biểu đồ cột hiển thị số lượng mẫu dương/âm trong tập huấn luyện và kiểm tra sau khi cân bằng và chia)*

**V. KẾT QUẢ THÍ NGHIỆM**

Mục đích cơ bản của kết quả thí nghiệm là kiểm tra hiệu suất của các bộ phân loại được thực hiện bằng Spark MLlib như LR, NB và Linear SVC. Hiệu suất của tất cả các bộ phân loại được đo bằng cách sử dụng các số liệu đánh giá khác nhau từ ma trận nhầm lẫn (confusion matrix) được hiển thị trong Bảng 02.

*(Bảng 02: Kết quả của các bộ phân loại sử dụng các số liệu đánh giá khác nhau)*
| Models             | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 87.38    | 86.54     | 88.78  | 87.64    |
| Naïve Bayes        | 83.43    | 82.35     | 88.78  | 85.44    |
| LinearSVC          | 88.38    | 88.54     | 88.39  | 88.46    |

Hình 11 cho thấy độ chính xác của mỗi mô hình. *(Hình 11: Biểu đồ cột so sánh độ chính xác của 3 mô hình)*
Hình 12 cho thấy việc đánh giá dưới dạng biểu đồ cột. *(Hình 12: Biểu đồ cột so sánh Accuracy, Precision, Recall, F1-Score cho 3 mô hình)*

Hơn nữa, chúng tôi cũng tính toán thời gian tính toán trong Bảng 03 của dữ liệu huấn luyện và Bảng 04 thời gian tính toán của dữ liệu kiểm tra.

*(Bảng 3: So sánh thời gian chạy của tập dữ liệu huấn luyện tính bằng giây)*
| Dataset | Logistic Regression | Naïve Bayes | LinearSVC |
|---------|---------------------|-------------|-----------|
| 130995  | 40.2                | 20.65       | 346.2     |

*(Bảng 4: So sánh thời gian chạy của tập dữ liệu kiểm tra tính bằng giây)*
| Dataset | Logistic Regression | Naïve Bayes | LinearSVC |
|---------|---------------------|-------------|-----------|
| 33019   | 0.06                | 0.06        | 0.15      |

**VI. KẾT LUẬN**

Phân tích tình cảm là một quá trình xác định cảm xúc của mọi người từ dữ liệu văn bản và thường được sử dụng trong kinh doanh để cải thiện chất lượng sản phẩm hoặc dịch vụ. Trong bài viết này, phân tích tình cảm (SA) được thực hiện trên bộ dữ liệu Đánh giá Fine Food của Amazon trong bối cảnh dữ liệu lớn. Các thử nghiệm khác nhau đã được thực hiện để phân tích tình cảm có bộ dữ liệu lớn bằng cách áp dụng các kỹ thuật phân loại khác nhau như NB, Linear SVC và LR bằng cách sử dụng Spark MLlib. Đối với bộ dữ liệu lớn, Apache Spark MLlib được sử dụng. Đối với thử nghiệm, 131186 đánh giá được sử dụng trong huấn luyện mô hình và 32829 đánh giá được sử dụng để kiểm tra mô hình. Một vài bước đã được áp dụng để khám phá và phân tích dữ liệu. Các thuật toán như Naïve Bayes, LinearSVC và hồi quy logistic đã được áp dụng. Bằng cách thực hiện phân tích, nó cho thấy rằng bộ phân loại linear support vector hoạt động tốt hơn các bộ phân loại khác. Trong tương lai, để cải thiện hiệu suất của bộ phân loại, các bộ đặc trưng khác nhau sẽ được xem xét như bi-gram, tri-gram và four-gram.

---

**Phân tích bài báo**

1.  **Mục tiêu:** Bài báo nhằm mục đích áp dụng các kỹ thuật phân tích dữ liệu lớn, cụ thể là Apache Spark và thư viện MLlib của nó, để thực hiện phân tích tình cảm (phân loại thành tích cực/tiêu cực) trên một bộ dữ liệu lớn các đánh giá thực phẩm trực tuyến từ Amazon. Đồng thời, so sánh hiệu suất của ba thuật toán học máy phổ biến: Linear SVC, Logistic Regression và Naïve Bayes trong bối cảnh này.

2.  **Vấn đề:** Lượng dữ liệu đánh giá trực tuyến ngày càng tăng (Big Data) đặt ra thách thức cho các hệ thống xử lý và phân tích truyền thống. Cần các công cụ và kỹ thuật có khả năng mở rộng để xử lý và trích xuất thông tin hữu ích (tình cảm của khách hàng) từ dữ liệu này một cách hiệu quả.

3.  **Phương pháp luận:**
    *   **Công cụ:** Sử dụng Apache Spark và thư viện MLlib, là những công cụ phù hợp cho xử lý dữ liệu lớn và học máy phân tán.
    *   **Dữ liệu:** Sử dụng bộ dữ liệu công khai Amazon Fine Food Reviews, một tập dữ liệu lớn và thực tế.
    *   **Quy trình:** Thực hiện một quy trình chuẩn trong học máy và xử lý ngôn ngữ tự nhiên:
        *   *Tiền xử lý:* Làm sạch dữ liệu (loại bỏ null/trùng lặp, nhiễu), chuẩn hóa điểm số thành nhãn nhị phân (tích cực/tiêu cực), tokenize và loại bỏ stop words. Việc chuẩn hóa điểm số (1-5) thành nhãn nhị phân (0/1) dựa trên "helpfulness" và sau đó cân bằng lớp là một bước quan trọng nhưng cũng có thể làm mất thông tin từ các điểm số trung gian.
        *   *Trích xuất đặc trưng:* Sử dụng TF-IDF để chuyển đổi văn bản thành vector số, một kỹ thuật phổ biến và hiệu quả cho dữ liệu văn bản.
        *   *Mô hình hóa:* Huấn luyện ba mô hình phân loại: Linear SVC, Logistic Regression, Naïve Bayes.
        *   *Đánh giá:* Sử dụng các số liệu tiêu chuẩn (Accuracy, Precision, Recall, F1-Score) và thời gian tính toán (huấn luyện và kiểm tra) trên tập dữ liệu đã được chia tách (80% train, 20% test) và cân bằng lớp.

4.  **Kết quả chính:**
    *   Cả ba mô hình đều đạt độ chính xác trên 80%, cho thấy tính khả thi của việc sử dụng Spark MLlib và các thuật toán này cho bài toán.
    *   LinearSVC đạt kết quả tốt nhất về các chỉ số Accuracy, Precision và F1-Score (Accuracy: 88.38%, F1: 88.46%).
    *   Naïve Bayes có thời gian huấn luyện nhanh nhất (20.65s), nhưng độ chính xác thấp nhất (83.43%).
    *   LinearSVC có thời gian huấn luyện chậm nhất (346.2s), nhưng lại cho kết quả đánh giá tốt nhất.
    *   Thời gian kiểm tra (dự đoán) rất nhanh cho cả Logistic Regression và Naïve Bayes (0.06s), và tương đối nhanh cho LinearSVC (0.15s).

5.  **Đóng góp:**
    *   Chứng minh việc áp dụng thành công Apache Spark và MLlib để phân tích tình cảm trên một bộ dữ liệu lớn trong thực tế (Amazon Food Reviews).
    *   Cung cấp một so sánh hiệu suất định lượng (cả về độ chính xác và thời gian) của ba thuật toán phân loại phổ biến (LinearSVC, Logistic Regression, Naïve Bayes) trong môi trường Big Data với Spark.
    *   Xác nhận LinearSVC là một lựa chọn mạnh mẽ cho loại bài toán này về mặt độ chính xác, mặc dù có chi phí thời gian huấn luyện cao hơn.

6.  **Hạn chế và Điểm cần lưu ý:**
    *   **Khẳng định về hiệu quả:** Bài báo kết luận LinearSVC "hoạt động hiệu quả hơn" (performs efficiently than) NB và LR. Tuy nhiên, dữ liệu thời gian chạy cho thấy NB huấn luyện nhanh nhất và LSVC chậm nhất. Có thể "hiệu quả" ở đây được đánh giá dựa trên sự cân bằng giữa độ chính xác và tài nguyên, hoặc chỉ tập trung vào độ chính xác cao nhất đạt được, nhưng điều này không được làm rõ hoàn toàn và có thể gây hiểu lầm nếu chỉ nhìn vào tốc độ.
    *   **Đơn giản hóa nhãn:** Việc chuyển đổi điểm số 1-5 thành nhãn nhị phân (0/1) dựa trên "helpfulness" và cân bằng lớp (bằng cách giảm mẫu lớp đa số) là một sự đơn giản hóa. Điều này có thể bỏ qua các sắc thái tình cảm (ví dụ: trung tính, hơi tích cực/tiêu cực) và làm mất thông tin từ việc giảm mẫu.
    *   **Đặc trưng cơ bản:** Nghiên cứu chỉ sử dụng TF-IDF dựa trên unigram (từ đơn). Việc sử dụng n-grams (bi-gram, tri-gram) như đề xuất trong phần kết luận có thể cải thiện hiệu suất bằng cách nắm bắt ngữ cảnh tốt hơn. Các đặc trưng khác (ví dụ: POS tagging, phân tích cú pháp) cũng không được khám phá.
    *   **Tính tổng quát:** Kết quả thu được trên bộ dữ liệu Amazon Food Reviews. Mức độ tổng quát hóa cho các loại đánh giá hoặc ngôn ngữ khác cần được kiểm chứng thêm.

7.  **Hướng phát triển tương lai (đề xuất bởi tác giả):** Sử dụng các bộ đặc trưng phức tạp hơn như bi-gram, tri-gram, four-gram để cải thiện hiệu suất mô hình.


