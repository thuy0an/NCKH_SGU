# Sentiment Classification for AFFR using PySpark

## Thông tin bài báo
- **Tác giả**: T.R. Aravidan, Vigneshwar C.N., Suganeshwari Gopalswamy  
- **Năm**: 2023  
- **Nguồn**: Recent Developments in Electronics and Communication Systems  
- **Mô hình**: Các thuật toán Machine Learning với PySpark  
- **Dataset**: Amazon Fine Food Reviews  
- **Link**: [Sentiment Classification for AFFR using PySpark](https://www.researchgate.net/publication/367070089_Sentiment_Classification_for_Amazon_Fine_Foods_Reviews_Using_Pyspark)  

## Hiệu suất mô hình
| Thuật toán        | Accuracy | Precision  | Recall | F1-Score  |   
|------------------|----------|--------|  ----------|--------| 
| Logistic Regression | 87.61 | 86.48 | 88.21 | 87.23 |  
| Naive Bayes          | 84.92 | 81.27 | 83.92 | 82.67 |

## Kết quả chính
- Sử dụng PySpark để huấn luyện mô hình phân loại sentiment.  
- Dữ liệu được tiền xử lý: loại bỏ trung lặp (3), loại bỏ các cột không cần thiết
- Áp dụng nhiều thuật toán ML như Logistic Regression, Decision Tree, Random Forest, Naive Bayes.  
- Sử dụng BoW và TF-IDF để trích xuất đặc trưng
- Kết quả với độ chính xác cao, khả năng của LR tốt hơn NB 

## Hạn chế
- Chưa kiểm thử trên tập dữ liệu khác.  
- hiệu suất phụ thuộc vào Spark cluster
Okay, đây là bản dịch và phân tích chi tiết bài báo khoa học thứ hai: "Sentiment Classification for Amazon Fine Foods Reviews Using Pyspark".

---

**Phân loại Tình cảm (Sentiment Classification) cho Đánh giá Amazon Fine Foods Sử dụng Pyspark**

**Tóm tắt:** Nghiên cứu cơ học về ý kiến, tình cảm, thái độ và cảm xúc của con người khi chúng được thể hiện qua giao tiếp được gọi là phân tích tình cảm hoặc khai thác ý kiến (opinion mining). Gần đây, đây là một trong những chủ đề phân tích phổ biến nhất trong khai thác văn bản (text mining) và Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing). Sự phổ biến rộng rãi của nó là do hai yếu tố: Yếu tố thứ nhất là ý kiến ​​là nền tảng cho thực tế mọi nỗ lực của con người và là yếu tố quyết định quan trọng đến hành vi của một người. Nó có một loạt các ứng dụng. Thứ hai, nó giải quyết một số vấn đề phân tích đầy thách thức mà trước đây chưa từng được thử nghiệm. Chúng tôi đã sử dụng Khung Dữ liệu Lớn (Big Data Framework) để giải quyết vấn đề vì các giải pháp truyền thống không thể xử lý sự tăng trưởng theo cấp số nhân của dữ liệu thực phẩm của Amazon. Trong nghiên cứu này, chúng tôi khảo sát các cách tiếp cận khác nhau để phân tích tình cảm cho các bộ dữ liệu lớn về đánh giá Amazon Fine Food bằng cách sử dụng công cụ xử lý dữ liệu Apache Spark và sử dụng MLlib, gói học máy cho Apache Spark, ba kỹ thuật—Linear SVC, Logistic Regression và Naive Bayes—với tỷ lệ chính xác hơn 80%. Chúng tôi nhận thấy rằng linear SVC vượt trội hơn NB và logistic regression. Thông qua phân tích thực nghiệm, chúng tôi nhận thấy rằng phân tích tình cảm thông qua một khung dữ liệu lớn hiệu quả hơn về thời gian tính toán so với phân tích tình cảm không có khung map-reduce.

**Từ khóa:** Dữ liệu lớn, Đánh giá Amazon Food, Xử lý ngôn ngữ tự nhiên, Hadoop, Spark.

**1. Giới thiệu**

Dữ liệu Lớn (Big Data) được mô tả là dữ liệu đa dạng, đến với tốc độ nhanh hơn và với khối lượng lớn hơn. Không có công cụ thao tác dữ liệu thông thường nào có thể lưu trữ hoặc phân tích Dữ liệu này một cách hiệu quả do kích thước và độ phức tạp khổng lồ của nó [1, 2]. Dữ liệu được tạo ra bởi các đánh giá của khách hàng là rất lớn, và với lượng dữ liệu khổng lồ này, không thể xử lý dữ liệu bằng các phương pháp thông thường trong python. Do đó, dữ liệu lớn là một công cụ quan trọng có thể làm cho quy trình hiệu quả và tiết kiệm đáng kể thời gian của chúng ta. Chúng ta có thể kiểm tra thông tin chủ quan của văn bản bằng phân loại tình cảm và khai thác ý kiến từ đó. Phân tích tình cảm thu thập thông tin từ nhận thức của mọi người về các thực thể, sự kiện, thuộc tính, quan điểm, đánh giá và cảm xúc khác nhau. Sự dễ dàng mà người mua đưa ra phán đoán khi tìm kiếm và lựa chọn các sự kiện, sản phẩm và thực thể trực tuyến bị ảnh hưởng đáng kể bởi ý kiến ​​của người khác. Các phương pháp phân tích tình cảm văn bản thường hoạt động ở một cấp độ cụ thể, chẳng hạn như cấp độ cụm từ, câu hoặc tài liệu [3].

Công trình hiện tại có thể hỗ trợ các doanh nghiệp hiểu rõ hơn những gì khách hàng đang nói về ẩm thực của họ và các nhiệm vụ khác như hệ thống đề xuất (recommender systems). Nó cũng có thể giúp khách hàng nhận ra nhà hàng nào tốt hơn những nhà hàng khác. Đại học California đã giới thiệu Apache Spark vào năm 2009. Nó có thể được sử dụng chủ yếu cho các bộ dữ liệu lớn, nhưng nó cũng hoạt động tốt với dữ liệu batch và streaming và có thể xử lý API trên các bộ dữ liệu khổng lồ [4]. Không giống như các tối ưu hóa khác như Hadoop, Spark là framework hiệu quả nhất cho dữ liệu lớn và cung cấp hiệu suất tuyệt vời. Thư viện có khả năng mở rộng Spark MLlib có thể được sử dụng với các ngôn ngữ lập trình cấp cao khác nhau. Các vấn đề học máy khác nhau được giải quyết thông qua phân tích tình cảm [5]. Chúng tôi đã sử dụng Spark MLlib trong công trình này vì nó là một thư viện hiện đại. MLlib của Spark đã được sử dụng cho nghiên cứu và có thể được sử dụng để phân tích dữ liệu lớn.

Spark MLlib cho dữ liệu kích thước lớn chỉ nhận được một lượng nghiên cứu rất nhỏ, nhưng lĩnh vực này vẫn cần rất nhiều sự phát triển. Công trình này khám phá các phương pháp mới để phân tích dữ liệu lớn nhằm đạt được hiệu suất và hiệu quả cao bằng cách sử dụng các kỹ thuật và phương pháp luận khác nhau trên MLlib của Spark.

**2. Tổng quan Nghiên cứu (Literature Survey)**

*   **2.1. Nghiên cứu về đánh giá mức độ hữu ích của đánh giá:** Các nghiên cứu trước đây về tính hữu ích của đánh giá chủ yếu tập trung vào việc tự động hóa dự đoán tính hữu ích của đánh giá để khắc phục các vấn đề như số lượng lớn và chất lượng không nhất quán của đánh giá khách hàng trực tuyến. Mối liên hệ giữa tính hữu ích của đánh giá [6] và các thuộc tính dựa trên văn bản đã được nghiên cứu rộng rãi bằng cách sử dụng các kỹ thuật khai thác văn bản và mô hình hóa thống kê. Tác động của tình cảm đến giá trị của đánh giá người dùng trực tuyến đã được kiểm tra trong nghiên cứu của Salehan et al. [3]. Nghiên cứu đã kiểm tra tác động của cực tính tình cảm (sentiment polarity) đến tính hữu ích của đánh giá trực tuyến và đưa ra một số phát hiện thú vị. Cả các số liệu liên quan đến tiêu đề và liên quan đến đánh giá đều được kết hợp trong phương pháp nghiên cứu của họ [1]. Độ dài và tuổi thọ của các đánh giá là những thành phần chính khác của mô hình, ngoài cực tính và thái độ của các đánh giá [7]. Các đánh giá được sắp xếp theo "hữu ích nhất" thay vì "mới nhất" trên các trang web như Amazon. Việc thêm tuổi thọ làm một trong các tiêu chí của nghiên cứu là phù hợp. Khi so sánh với các đánh giá có cực tính tích cực hoặc tiêu cực, người ta phát hiện ra rằng các đánh giá có cực tính trung tính ảnh hưởng lớn đến tính hữu ích [2]. Một yếu tố quan trọng khác là độ dài của đánh giá, vì người ta mong đợi một đánh giá dài hơn sẽ có nhiều thông tin hơn.
    Họ đã sử dụng thông tin từ CNET Download.com; họ đã xem xét cách các đánh giá trên internet ảnh hưởng đến số phiếu bầu hữu ích mà chúng nhận được. ALC (Tiêu chuẩn Thông tin Akaike) và tỷ lệ Lift đã được sử dụng trong nghiên cứu, sử dụng mô hình Hồi quy Logistic Thứ tự (Ordinal Logistic Regression) để đo độ chính xác [3]. Các đặc điểm ngữ nghĩa quan trọng hơn các đặc điểm khác, và những người đánh giá có ý kiến ​​mạnh mẽ đã nhận được nhiều phiếu bầu ủng hộ hơn [7]. Một mô hình hồi quy phi tuyến tính đã được Liu et al. (2008) tạo ra bằng cách sử dụng các thuộc tính lấy từ bộ dữ liệu đánh giá phim IMDB. Chuyên môn của người đánh giá, văn phong, tính kịp thời, độ dài, cực tính và xếp hạng trung bình chung của tất cả các đánh giá đã được xem xét. Tỷ lệ người dùng đã bỏ phiếu về việc đánh giá có hữu ích hay không trên tổng số người dùng được dùng làm giá trị xấp xỉ của số liệu hữu ích [3,7]. Ba đặc điểm được xác định là quan trọng nhất là kinh nghiệm của người đánh giá, văn phong và tính kịp thời. Điểm MSE được sử dụng để định lượng các tham số này và kết hợp chúng vào mô hình để đánh giá cả hiệu quả riêng lẻ và kết hợp của chúng. Khi các đặc trưng được sử dụng, kết quả tốt nhất đã được tạo ra. Kim et al. tập trung vào việc hoàn thành đánh giá tức thì về tính hữu ích của đánh giá để cung cấp phản hồi kịp thời cho tác giả đánh giá [7,8]. Họ đã sử dụng mô hình hồi quy SVM để điều tra xem các yếu tố dựa trên văn bản khác nhau ảnh hưởng như thế nào đến tính hữu ích của đánh giá.

*   **2.2. Nghiên cứu về phân loại văn bản:** Ngoài tính hữu ích của đánh giá, phần này cung cấp một số cái nhìn sâu sắc về công việc khai thác văn bản và NLP đang được thực hiện trong lĩnh vực này. Phát hiện thư rác được phân loại là "spam" hoặc "không phải spam," các thuộc tính được truy xuất từ ​​văn bản bằng NLP đã được thảo luận và đối chiếu theo nhiều cách khác nhau. Nghiên cứu đã sử dụng nhiều biến từ vựng và ngữ nghĩa, chẳng hạn như Bag of Words, Term Frequency, POS tagging, và những biến khác. Các phương pháp Support Vector Machine, Nave Bayes và Logistic Regression đã được sử dụng để giải quyết vấn đề phân loại. Mặc dù đôi khi chúng tỏ ra vượt trội hơn giữa các thuật toán được thử nghiệm, SVM hoạt động tốt hơn Naive Bayes và Logistic Regression [9]. Hiệu suất cũng tăng lên khi nhiều đặc điểm được tích hợp. Một nhà nghiên cứu đã sử dụng các đặc điểm phong cách học (stylometric traits) để tách biệt các phong cách viết của người gửi thư rác nhằm xác định các ý kiến ​​sai lệch trong công việc của họ. Họ phát hiện ra rằng những người gửi thư rác cố gắng thay đổi giao tiếp bằng giọng nói và văn bản của họ bằng cách sử dụng các thuật ngữ ngắn hơn, đơn giản hơn và có ít âm tiết trung bình hơn trên mỗi từ. Những đặc điểm này bao gồm tổng số token, độ dài câu trung bình, độ dài token trung bình và tần suất của các ký tự viết hoa. Các phương pháp phân loại ML bao gồm Nave Bayes và Support Vector Machines (SVM) [9,10].

**3. Công việc đề xuất**

Nghiên cứu này sử dụng bộ dữ liệu của chúng tôi để áp dụng các ý tưởng PySpark nhằm hiểu rõ hơn về chúng. Thực hiện phân loại nhị phân bằng cách sử dụng dữ liệu dựa trên văn bản và các kỹ thuật phân loại học máy là vấn đề đang được xem xét. Các lớp nhị phân sẽ được cung cấp, với "1" biểu thị "Hữu ích" và "0" biểu thị "Không hữu ích." Số liệu về mức độ hữu ích sẽ dựa trên tỷ lệ phần trăm tổng thể của những người đánh dấu bài đánh giá là "hữu ích." Nghiên cứu này sẽ sử dụng nhiều loại đặc trưng dựa trên văn bản, chẳng hạn như đặc trưng vector hóa dựa trên ma trận, đặc trưng dựa trên nhúng từ (word embedding) và các đặc trưng thu thập từ văn bản đánh giá và tóm tắt, chẳng hạn như phân tích cấu trúc, cú pháp và ngữ nghĩa. Đánh giá sản phẩm, tóm tắt đánh giá, xếp hạng đánh giá, thông tin về phiếu bầu hữu ích và dữ liệu khác liên quan đến sản phẩm và người dùng sẽ được sử dụng để tạo mô hình dự đoán.

*   **3.1. Bộ dữ liệu:** Bộ dữ liệu này chứa các đánh giá của Amazon về các bữa ăn tinh tế. Thông tin bao gồm tất cả 500.000 đánh giá cho đến tháng 10 năm 2012 và kéo dài hơn mười năm. Các đánh giá bao gồm văn bản thuần túy, thông tin người dùng và sản phẩm, xếp hạng và đánh giá. Các đánh giá từ tất cả các danh mục Amazon khác cũng được bao gồm. Dữ liệu đánh giá có sẵn từ tháng 10 năm 1999 đến tháng 10 năm 2012. Hình 1 mô tả các giai đoạn khác nhau của công việc được đề xuất.

*   **3.2. Kiến trúc Hệ thống:** Hình 1 mô tả các giai đoạn khác nhau của công việc được đề xuất.
    *(Hình 1: Sơ đồ Kiến trúc: Amazon Dataset -> Text Preparation -> Text Preprocessing -> Model Building -> Evaluation)*

**4. Công việc đề xuất (Chi tiết)**

*   **4.1. Chuẩn bị Dữ liệu:** Bộ dữ liệu chúng tôi đang sử dụng là bộ dữ liệu đánh giá amazon fine food bao gồm các đánh giá về các nhà hàng khác nhau dựa trên phản hồi của khách hàng. Sau đó, chúng tôi cố gắng hiểu dữ liệu bằng cách loại bỏ các cột không cần thiết và lọc tất cả các đánh giá trung tính (neural reviews). Chúng tôi loại bỏ tất cả các đánh giá có điểm số là ba và gán các đánh giá có điểm số lớn hơn ba là tích cực và nhỏ hơn ba là tiêu cực. Trong công trình này, chúng tôi gán một xếp hạng nhị phân - đánh giá tích cực là một và đánh giá tiêu cực là 0.

*   **4.2. Tiền xử lý Văn bản:** Giai đoạn này bao gồm hai giai đoạn nữa như được mô tả dưới đây:
    *   Làm sạch dữ liệu (Data Cleaning):
    *   Gắn thẻ và Chuẩn hóa gốc từ (Tagging and Lemmatization)

    *(a) Vector hóa Bag of Words (BoW):* Một phương pháp để trích xuất các đặc trưng từ văn bản để sử dụng trong mô hình hóa, chẳng hạn như các kỹ thuật học máy, được gọi là mô hình bag-of-words (BoW). Hãy xem xét hai đánh giá sau làm ví dụ:
    r1 = {"Thức ăn tuyệt vời, không gian tuyệt vời"} và r2 = {"Tôi yêu món ăn này"}
    r1' = {"The", "food", "is", "great", "ambience", "is", "great"} và r2' = {"I", "love", "this", "food"}
    -> V = {"The", "food", "is", "great", "ambience", "I", "love", "this"}
    Khi chúng tôi sử dụng vector 'V' được tạo ở trên, biểu diễn vector cho mỗi đánh giá r1 và r2 sẽ trông giống như thế này r1 vector = [1,1,2,2,1,0,0,0] r2 vector = [0,1,0,0,1,1,1,1]. Chúng tôi đặt số đếm thành hai vì các cụm từ "great" và "is" xuất hiện hai lần trong r1. Khi một từ không có trong một đánh giá, số đếm sẽ được đặt thành 0. Ví dụ trên được thiết kế để giúp bạn hiểu cách hoạt động của Bag of words ngay cả khi "is" được coi là một stopword.

    *(B) Vector hóa TF-IDF:* TFIDF, được gọi là tần số thuật ngữ-tần số tài liệu nghịch đảo (term frequency-inverse document frequency), là một thống kê số được sử dụng trong truy xuất thông tin nhằm mục đích truyền đạt tầm quan trọng của một từ đối với một tài liệu trong một bộ sưu tập.

**5. Xây dựng Mô hình**

Chúng tôi đã sử dụng mô hình Logistic Regression và naive Bayes
*   **Logistic Regression:** Mô hình hồi quy logistic phổ biến nhất trong ví dụ của chúng tôi dự đoán một kết quả nhị phân—đánh giá thuận lợi hoặc không thuận lợi.
*   **Naive Bayes:** Một kỹ thuật phân loại được gọi là Naive Bayes dựa trên Định lý Bayes.

**6. Phân tích Thí nghiệm**

Sau khi gắn thẻ, chúng tôi đã tokenize, tách và lưu trữ từng từ riêng biệt đồng thời loại bỏ mọi stop words. Chúng tôi chuyển đổi cặp khóa/giá trị hiện tại của mình để tính toán tần suất từ ​​thành một cặp khóa-giá trị mới với document-id và token làm khóa và 1 (biểu thị số đếm) làm giá trị. Ở đây, chúng tôi sẽ sử dụng flatMap() để nhóm tất cả các token vào một danh sách duy nhất. Chúng tôi sẽ tổng hợp các giá trị cho cùng một khóa và nhóm các cặp khóa/giá trị với khóa chung để nhận tần số thuật ngữ (term frequency) cho một từ cụ thể tương ứng với document-id của nó như trong Hình 2. Chúng tôi đang thay đổi các cặp khóa-giá trị thành một tập hợp cặp khóa-giá trị mới với token làm khóa và document-id của nó và tần số thuật ngữ tương ứng làm giá trị trong Hình 3.
*(Hình 2: cặp khóa-giá trị (document-id, 1) - Ví dụ: ((2, 'coat'), 1))*
*(Hình 3: (token, (document-id, term frequency)) - Ví dụ: [('several', (0, 1))]*)

Chúng tôi đã ánh xạ cặp khóa-giá trị trước đó sang một cặp khóa-giá trị mới để tính toán Tần số Tài liệu Nghịch đảo (Inverse Document Frequency - IDF). Khóa sẽ là token, và giá trị của nó sẽ là document id TF cho token đó cùng với một bộ đếm là 1. Số một ở đây cho biết sự hiện diện của một từ trong tài liệu được liên kết với nó, như trong Hình 4.
*(Hình 4: Cặp khóa-giá trị mới với document-id, token làm khóa và 1 (biểu thị số đếm) làm giá trị - Ví dụ: ('gelatin', (2, 1, 1)))*

Chúng tôi đã trích xuất token và số đếm là 1, đại diện cho sự xuất hiện của nó trong các tài liệu nhất định trong Hình 5. Vì chúng tôi có số lượng tài liệu chứa mỗi token w, chúng tôi vừa ánh xạ đầu ra cuối cùng này với phép biến đổi logarit để tính điểm IDF trong Hình 6.
*(Hình 5: Trích xuất token - Ví dụ: [('several', 1)])*
*(Hình 6: Các từ với TF-IDF - Ví dụ: ('finicky', 1.0))*

**7. Kết quả**

Bảng 1 cho thấy hiệu suất của các mô hình Logistic Regression và Naïve Bayes sử dụng các số liệu đánh giá khác nhau.

*(Bảng 1: Kết quả của các mô hình sử dụng các số liệu đánh giá khác nhau)*
| Models             | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 87.61    | 86.48     | 88.21  | 87.23    |
| Naïve Bayes        | 84.92    | 81.27     | 83.92  | 82.67    |

**8. Kết luận**

Phân tích tình cảm được sử dụng trong kinh doanh để tăng chất lượng hàng hóa và dịch vụ bằng cách xác định cảm xúc của mọi người từ dữ liệu văn bản. Bộ dữ liệu Fine Food Reviews từ Amazon được sử dụng trong việc triển khai phân tích tình cảm của bài viết này trên một tập dữ liệu lớn. Là một phần của các trình bày khác nhau sử dụng Spark MLlib, các thuật toán phân loại như Naïve Bayes và Logistic regression đã được áp dụng để phân tích cảm xúc trong các bộ dữ liệu khổng lồ. Apache Spark MLlib được sử dụng cho các bộ dữ liệu khổng lồ. Mười nghìn đánh giá được sử dụng để đánh giá các mô hình, trong khi 40000 đánh giá được sử dụng để huấn luyện các mô hình cho thử nghiệm. Dữ liệu đã được khám phá và phân tích bằng một vài quy trình đơn giản. Công việc được đề xuất trong nền tảng MapReduce vượt trội hơn cùng một thuật toán học máy không có MapReduce. Trong nghiên cứu sắp tới của chúng tôi, chúng tôi sẽ tạo một từ điển miền (domain thesaurus) để biểu diễn sở thích của người dùng, và sau đó chúng tôi sẽ sử dụng lọc cộng tác dựa trên người dùng (user-based collaborative filtering) để đề xuất các mặt hàng thực phẩm dựa trên các đánh giá.

---

**Phân tích bài báo**

1.  **Mục tiêu:** Bài báo này cũng nhằm mục đích thực hiện phân loại tình cảm trên bộ dữ liệu Amazon Fine Food Reviews bằng cách sử dụng Pyspark (giao diện Python cho Spark) và thư viện MLlib. Mục tiêu cụ thể là so sánh hiệu suất của Logistic Regression (LR) và Naïve Bayes (NB) và nhấn mạnh lợi ích về tốc độ tính toán khi sử dụng khung dữ liệu lớn (ám chỉ MapReduce/Spark) so với các phương pháp không sử dụng khung này.

2.  **Vấn đề:** Tương tự như bài báo trước, vấn đề cốt lõi là sự tăng trưởng nhanh chóng của dữ liệu đánh giá trực tuyến, vượt quá khả năng xử lý của các phương pháp truyền thống (ví dụ: Python đơn thuần được đề cập).

3.  **Phương pháp luận:**
    *   **Công cụ:** Pyspark, Spark MLlib.
    *   **Dữ liệu:** Amazon Fine Food Reviews. Tuy nhiên, cách xử lý dữ liệu và kích thước sử dụng khác biệt:
        *   *Chuẩn bị dữ liệu:* Loại bỏ các đánh giá có điểm số = 3. Chuyển đổi điểm số thành nhãn nhị phân: điểm < 3 là 0 (tiêu cực), điểm > 3 là 1 (tích cực).
        *   *Kích thước:* Sử dụng 40.000 đánh giá để huấn luyện và 10.000 đánh giá để kiểm tra (tổng 50.000), nhỏ hơn đáng kể so với bài báo đầu tiên và là một tập con của bộ dữ liệu gốc (>500k).
    *   **Quy trình:**
        *   *Tiền xử lý:* Đề cập "Data Cleaning", "Tagging and Lemmatization" nhưng chỉ giải thích chi tiết về Bag-of-Words (BoW) và TF-IDF. Phần Phân tích Thí nghiệm mô tả các bước tính toán TF-IDF theo kiểu MapReduce (flatMap, aggregate/reduceByKey).
        *   *Trích xuất đặc trưng:* BoW và TF-IDF.
        *   *Mô hình hóa:* Chỉ huấn luyện và đánh giá Logistic Regression và Naïve Bayes. **Linear SVC được đề cập nhiều lần trong Tóm tắt và Giới thiệu nhưng không được triển khai hoặc báo cáo kết quả.**
        *   *Đánh giá:* Sử dụng Accuracy, Precision, Recall, F1-Score.

4.  **Kết quả chính:**
    *   Logistic Regression hoạt động tốt hơn một chút so với Naïve Bayes trên tất cả các chỉ số (ví dụ: Accuracy 87.61% vs 84.92%).
    *   Không có kết quả nào được trình bày cho Linear SVC.
    *   Bài báo khẳng định phương pháp tiếp cận dựa trên khung dữ liệu lớn (MapReduce/Spark) hiệu quả hơn về thời gian tính toán so với không dùng, nhưng không cung cấp bằng chứng so sánh thời gian chạy trực tiếp trong phần kết quả.

5.  **Đóng góp:**
    *   Cung cấp một ví dụ khác về việc áp dụng Pyspark/MLlib cho phân tích tình cảm trên tập dữ liệu Amazon.
    *   Mô tả chi tiết các bước tính toán TF-IDF theo logic MapReduce.
    *   So sánh LR và NB trong thiết lập cụ thể này (với cách chuẩn bị dữ liệu và kích thước tập con khác biệt).

6.  **Hạn chế và Điểm cần lưu ý:**
    *   **Mâu thuẫn nghiêm trọng:** Điểm yếu lớn nhất là sự mâu thuẫn giữa Tóm tắt/Giới thiệu (nói rằng Linear SVC được sử dụng và hoạt động tốt nhất) và phần Phương pháp/Kết quả (chỉ trình bày LR và NB, không có LSVC). Điều này làm giảm đáng kể độ tin cậy của bài báo.
    *   **Thiếu bằng chứng thực nghiệm:** Tuyên bố về lợi thế tốc độ của khung Big Data/MapReduce không được chứng minh bằng dữ liệu so sánh thời gian chạy cụ thể trong bài báo.
    *   **Xử lý dữ liệu:** Việc loại bỏ hoàn toàn các đánh giá có điểm 3 và sử dụng một tập con nhỏ hơn (50k so với >160k ở bài 1) có thể ảnh hưởng đến tính tổng quát của kết quả.
    *   **Chi tiết tiền xử lý:** Thiếu mô tả chi tiết về "Data Cleaning", "Tagging", và "Lemmatization" đã thực sự được thực hiện như thế nào.

7.  **Hướng phát triển tương lai (đề xuất bởi tác giả):** Xây dựng từ điển miền (domain thesaurus) và sử dụng lọc cộng tác dựa trên người dùng để xây dựng hệ thống đề xuất thực phẩm, chuyển hướng khỏi mục tiêu phân loại tình cảm ban đầu.



