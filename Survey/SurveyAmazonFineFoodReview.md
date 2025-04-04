# Survey for Amazon Fine Food Reviews

## Thông tin bài báo

\- Tác giả: Vijay Bhati, Jayveer Kher

\- Năm: 2019

\- Nguồn: IRJET (International Research Journal of Engineering and Technology)

\- Mô hình: Naïve Bayes, SVM (Support Vector Machine), K-Means Clustering

\- Dataset: Amazon Fine Food Reviews

\- Link: [Survey for Amazon Fine Food Reviews](https://scholar.google.com/scholar?hl=vi&as_sdt=0%2C5&q=Survey+for+Amazon+Fine+Food+Reviews&btnG=)

## Hiệu suất mô hình

\- Logistic Regression provides the best sentimental analysis result.

## Kết quả chính

\- Sử dụng mô hình BERT để phân tích sentiment từ các bài đánh giá thực
phẩm trên Amazon.

\- Tiền xử lý dữ liệu bao gồm loại bỏ dữ liệu không cần thiết, xử lý dữ
liệu trùng lặp và chuyển đổi văn bản sang định dạng phù hợp.

\- Chia tập dữ liệu thành các nhóm: huấn luyện, kiểm tra và xác thực.

\- Áp dụng kỹ thuật fine-tuning trên BERT để cải thiện hiệu suất mô
hình.

## Hạn chế

\- Không có sự so sánh trực tiếp với các mô hình khác.

\- Chưa kiểm thử trên các tập dữ liệu khác để đánh giá độ tổng quát.



---


**Tóm tắt** - Trong kỷ nguyên của mạng xã hội và internet, phân tích tình cảm cực kỳ hữu ích trong việc giám sát mạng xã hội vì nó cho phép chúng ta có cái nhìn tổng quan về dư luận rộng rãi hơn đằng sau các chủ đề nhất định. Nghiên cứu của chúng tôi tập trung vào việc tạo ra mô hình dự đoán nơi chúng tôi có thể dự đoán liệu một đề xuất là tích cực hay tiêu cực. Trong phân tích này, chúng tôi sẽ tập trung vào điểm số cũng như tình cảm tích cực/tiêu cực của đề xuất. Phân tích chứng minh rằng thuật toán hồi quy logistic cung cấp kết quả phân tích tình cảm tốt nhất. Kết quả của chúng tôi được xác minh thêm bởi độ chính xác đáng kinh ngạc của bộ phân loại hồi quy Logistic.

**Từ khóa** - Học máy, Xử lý Ngôn ngữ Tự nhiên, Học sâu, Khai thác Văn bản, Phân tích Tình cảm, Phân tích Dữ liệu, Trí tuệ Nhân tạo.

**I. GIỚI THIỆU**

Như với tất cả các lĩnh vực của học máy, sự đổi mới xung quanh phân tích tình cảm đang diễn ra với tốc độ chóng mặt và phạm vi sử dụng của nó là rất lớn. Là một công cụ cực kỳ có giá trị cho các công ty truyền thông xã hội, chủ doanh nghiệp và nhà quảng cáo, phân tích tình cảm đã cung cấp những hiểu biết sâu sắc giúp thúc đẩy các quyết định kinh doanh, chiến lược và mục tiêu hiệu quả trên nhiều lĩnh vực. Những hiểu biết này bao gồm từ phân tích đánh giá về thương hiệu của bạn và đối thủ cạnh tranh đến so sánh sự đón nhận sản phẩm của bạn ở các thị trường quốc tế mới.

Bài báo này nhằm mục đích phân tích một giải pháp cho việc phân loại tình cảm ở mức độ chi tiết, cụ thể là mức câu, trong đó cực tính của câu có thể được đưa ra bởi ba loại là tích cực, tiêu cực và trung lập. Trong công việc này, mục tiêu là dự đoán điểm số của các bài đánh giá thực phẩm trên thang điểm từ 1 đến 5 bằng hai mạng nơ-ron hồi quy (recurrent neural networks - RNN) được tinh chỉnh cẩn thận. Đối với mô hình cơ sở (baseline), chúng tôi huấn luyện một RNN đơn giản để phân loại. Sau đó, chúng tôi mở rộng mô hình cơ sở thành RNN sửa đổi và GRU. Ngoài ra, chúng tôi trình bày hai phương pháp khác nhau để đối phó với dữ liệu bị lệch nhiều (highly skewed data), một vấn đề phổ biến đối với các bài đánh giá.

Các mô hình được đánh giá bằng độ chính xác (accuracies). Chúng tôi đã xem xét một microblog phổ biến như vậy được gọi là Đánh giá (Reviews) và xây dựng các mô hình để phân loại "đánh giá" thành tình cảm tích cực, tiêu cực. Chúng tôi xây dựng các mô hình cho hai nhiệm vụ phân loại: một nhiệm vụ nhị phân phân loại tình cảm thành các lớp tích cực và tiêu cực và một nhiệm vụ 3 chiều phân loại tình cảm thành các lớp tích cực, tiêu cực và trung lập. Chúng tôi thử nghiệm với ba loại mô hình: mô hình unigram, mô hình dựa trên đặc trưng (feature based model) và mô hình dựa trên hạt nhân cây (tree kernel based model). Đối với mô hình dựa trên đặc trưng, chúng tôi sử dụng một số đặc trưng được đề xuất trong các tài liệu trước đây và đề xuất các đặc trưng mới. Đối với mô hình dựa trên hạt nhân cây, chúng tôi thiết kế một biểu diễn cây mới cho các bài đánh giá. Chúng tôi sử dụng mô hình unigram, trước đây đã được chứng minh là hoạt động tốt cho phân tích tình cảm đối với dữ liệu đánh giá, làm mô hình cơ sở của chúng tôi. Các thử nghiệm của chúng tôi cho thấy rằng một mô hình unigram thực sự là một mô hình cơ sở khó vượt qua, đạt được hơn 20% so với đường cơ sở ngẫu nhiên (chance baseline) cho cả hai nhiệm vụ phân loại. Mô hình dựa trên đặc trưng của chúng tôi chỉ sử dụng 100 đặc trưng đạt được độ chính xác tương tự như mô hình unigram sử dụng hơn 10.000 đặc trưng. Mô hình dựa trên hạt nhân cây của chúng tôi vượt trội hơn cả hai mô hình này với một biên độ đáng kể.

Chúng tôi cũng thử nghiệm với sự kết hợp của các mô hình: kết hợp unigram với các đặc trưng của chúng tôi và kết hợp các đặc trưng của chúng tôi với hạt nhân cây. Cả hai sự kết hợp này đều vượt trội hơn mô hình cơ sở unigram hơn 4% cho cả hai nhiệm vụ phân loại. Trong bài báo này, chúng tôi trình bày phân tích đặc trưng sâu rộng về 100 đặc trưng mà chúng tôi đề xuất. Các thử nghiệm của chúng tôi cho thấy rằng các đặc trưng liên quan đến các đặc điểm cụ thể của đánh giá (biểu tượng cảm xúc, hashtag, v.v.) làm tăng giá trị cho bộ phân loại nhưng chỉ một cách cận biên. Các đặc trưng kết hợp cực tính trước đó của từ với các thẻ từ loại (parts-of-speech tags) của chúng là quan trọng nhất cho cả hai nhiệm vụ phân loại.

Do đó, chúng tôi thấy rằng các công cụ xử lý ngôn ngữ tự nhiên tiêu chuẩn rất hữu ích ngay cả trong một thể loại khá khác biệt so với thể loại mà chúng được đào tạo (newswire). Hơn nữa, chúng tôi cũng chỉ ra rằng mô hình hạt nhân cây hoạt động gần như tốt như các mô hình dựa trên đặc trưng tốt nhất, mặc dù nó không yêu cầu kỹ thuật đặc trưng chi tiết. Chúng tôi sử dụng dữ liệu đánh giá được chú thích thủ công cho các thử nghiệm của mình. Một lợi thế của dữ liệu này, so với các bộ dữ liệu đã sử dụng trước đây, là các bài đánh giá được thu thập theo kiểu luồng (streaming fashion) và do đó đại diện cho một mẫu thực sự của các bài đánh giá thực tế về mặt sử dụng ngôn ngữ và nội dung. Bộ dữ liệu mới của chúng tôi có sẵn cho các nhà nghiên cứu khác. Chúng tôi giới thiệu hai tài nguyên có sẵn, đó là từ điển chú thích thủ công cho các biểu tượng cảm xúc và từ điển từ viết tắt được thu thập từ web với bản dịch tiếng Anh của hơn 5000 từ viết tắt thường được sử dụng.

**II. TẬP DỮ LIỆU**

Tập dữ liệu được lấy từ Kaggle [1] và với một bài đánh giá cho trước, chúng tôi phải xác định xem bài đánh giá đó là tích cực (điểm 4 hoặc 5) hay tiêu cực (điểm 1 hoặc 2). Dữ liệu kéo dài hơn 16 năm, bao gồm tất cả ~500.000 bài đánh giá từ tháng 10 năm 1997 đến tháng 10 năm 2012. Các bài đánh giá bao gồm thông tin sản phẩm và người dùng, xếp hạng và một bài đánh giá văn bản thuần túy. Nó cũng bao gồm các đánh giá từ tất cả các danh mục Amazon khác. Tất cả dữ liệu nằm trong 2 tệp, Train và Test.
1.  Train.csv chứa 5 cột: ProductId, Time, Title, Summary, Text
2.  Test.csv chứa các cột tương tự mà chúng ta phải dự đoán.
3.  Kích thước của Train.csv: 6.75GB
4.  Kích thước của Test.csv: 2GB
5.  Số lượng hàng trong Train.csv = 568.454

**1. Dữ liệu bao gồm**
1.  Đánh giá từ tháng 10 năm 1997 - tháng 10 năm 2012
2.  568.454 đánh giá
3.  256.059 người dùng
4.  74.258 sản phẩm
5.  260 người dùng có > 50 đánh giá

**2. Thông tin thuộc tính**
1.  Id
2.  ProductId - định danh duy nhất cho sản phẩm
3.  UserId - định danh duy nhất cho người dùng
4.  ProfileName
5.  HelpfulnessNumerator - số lượng người dùng thấy đánh giá hữu ích
6.  HelpfulnessDenominator - số lượng người dùng cho biết họ có thấy đánh giá hữu ích hay không
7.  Score - xếp hạng từ 1 đến 5
8.  Time - dấu thời gian cho đánh giá
9.  Summary - tóm tắt ngắn gọn của đánh giá
10. Text - văn bản của đánh giá

**III. PHƯƠNG PHÁP ĐỀ XUẤT**

*(Hình 1: Đám mây từ của toàn bộ tập dữ liệu. Các từ nổi bật: flavor, product, taste, good, great, buy, love, candy, sweet, dog, treat...)*

*(Hình 2: Sơ đồ quy trình của phương pháp được sử dụng)*
1.  Hiểu bài toán.
2.  Sử dụng bộ dữ liệu đánh giá thực phẩm tinh chế của Amazon để lấy đánh giá của khách hàng.
3.  Thực hiện các bước tiền xử lý dữ liệu trên dữ liệu thô đã trích xuất như loại bỏ ký tự đặc biệt, từ ngắn, Tokenization, stemming, v.v.
4.  Chia dữ liệu thành hai phần: Dữ liệu huấn luyện và dữ liệu kiểm tra.
5.  (Nhánh Huấn luyện) Sử dụng Word cloud để xem các từ phổ biến nhất.
6.  Sử dụng các kỹ thuật phân tích tình cảm để tìm điểm cực tính cũng như tình cảm của các bài đánh giá.
7.  Sử dụng các kỹ thuật trực quan hóa dữ liệu để phân tích tình cảm phổ biến - tích cực, tiêu cực hoặc trung lập.
8.  Sử dụng kỹ thuật Bag of words, TF-IDF, word2vec để trích xuất đặc trưng.
9.  Áp dụng thuật toán Logistic regression, Naïve Bayes, knn, và decision Tree để kiểm tra độ chính xác của kết quả.
10. (Nhánh Kiểm tra) - Dữ liệu kiểm tra được đưa vào mô hình đã huấn luyện.
11. Kết quả.

**(Trang 3)**

*(Hình 3: Đám mây từ của các đánh giá tích cực. Các từ nổi bật: great, good, flavor, love, taste, product, buy, use, treat, dog...)*
*(Hình 4: Đám mây từ của các đánh giá tiêu cực. Các từ nổi bật: taste, product, flavor, dont, buy, good, bad, box, order, soup...)*
*(Hình 5: Biểu đồ phân phối Seaborn trên tập dữ liệu. Cho thấy sự phân bố của điểm số, tập trung cao ở điểm 5)*

**1. Tiền xử lý dữ liệu**
Tiền xử lý dữ liệu có thể là một bước rất quan trọng trong phân tích của chúng tôi vì nó có thể có tác động nghiêm trọng đến kết quả. Một tập dữ liệu chưa được xử lý có thể gây ra kết quả sai và có thể làm hỏng phân tích; do đó, cần phải tiền xử lý thông tin trước khi áp dụng bất kỳ hoạt động khai thác dữ liệu nào. Trong tiền xử lý dữ liệu, chúng tôi có xu hướng loại bỏ các thẻ không mong muốn, liên kết web và các ký hiệu đặc biệt (@ # ^ * "/ : >, < \ |?), có thể dẫn đến kết quả sai. Quy trình sau đã được tuân theo để xử lý dữ liệu:
1.  Loại bỏ dữ liệu trùng lặp.
2.  Xử lý Tên người dùng đánh giá (Removing Reviews Handles)
3.  Loại bỏ Dấu câu, Số, Ký tự đặc biệt
4.  Loại bỏ Từ dừng (Stop Words)
5.  Tokenization (Tách từ)
6.  Stemming (Rút gọn từ)

**2. Trích xuất đặc trưng**
Một vài kỹ thuật khác nhau đã được sử dụng trong bài báo này để trích xuất các đặc trưng, cụ thể là bag of words (túi từ), tf-idf và TNSE word2vec.
*   **Bag of words (BoW):** Một unigram của BoW đã được tạo ra với 2 perplexity khác nhau để xây dựng từ vựng cho các thuật toán học máy kiểm tra độ chính xác. Unigram với perplexity là 30 được hiển thị trong hình 6 và hình 7 với perplexity là 50.
    *(Hình 6: BoW Perplexity=30 - Biểu đồ t-SNE)*
    *(Hình 7: BoW Perplexity=50 - Biểu đồ t-SNE)*
*   Hơn nữa, một bigram của BoW cũng được tính toán cho cả dữ liệu kiểm tra và huấn luyện và đường cong AUC đã được vẽ. Phần còn lại được hiển thị trong hình 8.
    *(Hình 8: Đường cong AUC trên bigram BoW. Hiển thị AUC cho tập huấn luyện và tập kiểm tra)*
*   Sau đó trong bài báo này, độ chính xác của đặc trưng được trích xuất bằng BoW được hiển thị trên thuật toán cây quyết định.
*   **TF-IDF:** Tương tự như BoW, hai perplexity khác nhau là 20 & 30 với 10K điểm mỗi loại đã được vẽ đồ thị, được hiển thị trong Hình 9 và Hình 10 tương ứng. *(Chú thích hình ảnh trong bài báo có vẻ nhầm lẫn số thứ tự và nội dung so với mô tả này - Hình 8, 9 trong bài báo gốc là TF-IDF)*
    *(Hình 9 (theo bài báo): TFIDF với perplexity 20 - Biểu đồ t-SNE)*
    *(Hình 10 (theo bài báo): TFIDF với perplexity 30 - Biểu đồ t-SNE)*

**(Trang 4)**

*   **TNSE word2vec:** TNSE word to vector đã được sử dụng để tìm các tập hợp con tương tự của cùng một từ và đồ thị đã được vẽ, có thể thấy trong Hình 10 (theo bài báo là Hình 8).
    *(Hình 10 (theo bài báo): t-SNE trên word2vec. Các điểm được phân cụm nhưng có sự chồng chéo đáng kể giữa các lớp Tích cực và Tiêu cực)*

**Kết luận từ các biểu đồ TSNE:** Hầu hết các biểu đồ TSNE cho thấy dữ liệu khá chồng chéo, do đó chúng tôi không thể chắc chắn rằng dữ liệu có thể phân tách tuyến tính nhưng vì TSNE là một thuật toán xấp xỉ, chúng tôi không thể chắc chắn về khẳng định này. Do đó, chúng tôi cần tạo mô hình và tự kiểm tra. Nếu dữ liệu từ biểu đồ TSNE được nhìn thấy là có thể phân tách, nó sẽ dễ dàng phân tách bằng bất kỳ mô hình tuyến tính nào.

**IV. KẾT QUẢ**

Chúng tôi đã áp dụng bốn kỹ thuật vector hóa trên tập dữ liệu. Mục tiêu của việc sử dụng thuật toán Học máy là tạo ra một mô hình huấn luyện có thể được sử dụng để dự đoán lớp hoặc giá trị của biến mục tiêu bằng cách học các quy tắc quyết định đơn giản được suy ra từ dữ liệu trước đó (dữ liệu huấn luyện).
*   **BoW (Uni-gram, bi-gram) và TF-IDF:** Sẽ mất rất nhiều thời gian nếu lấy tất cả các chiều vì nó có kích thước rất lớn và do đó đã thử với max 300 là max_depth. Vector hóa Bi-gram (max_depth=73) cho kết quả tốt nhất với độ chính xác 85.11% và điểm F1 là 0.513.
*   **Tầm quan trọng đặc trưng:** Đã vẽ biểu đồ tầm quan trọng của đặc trưng cho Uni-gram, bi-gram và tfidf nhưng không phải cho Avg Word2Vec và Tfidf Avg Word2Vec vì các vector hóa Word2Vec có độ tương quan cao nên không thể trực tiếp lấy được tầm quan trọng của đặc trưng. Hình 11 hiển thị tầm quan trọng của đặc trưng trên BOW unigram bằng cây quyết định, Hình 12 trên bigram BOW và Hình 13 trên TF-IDF.
    *(Hình 11: Tầm quan trọng đặc trưng BOW unigram bằng cây quyết định)*
    *(Hình 12: Tầm quan trọng đặc trưng BOW bigram)*
    *(Hình 13: Tầm quan trọng đặc trưng trên tập dữ liệu (có vẻ là TF-IDF))*
*   **KNN:** Độ chính xác tốt nhất là 85.107% đạt được bằng vector hóa Avg. Word2Vec. Việc triển khai kd-tree và brute của KNN cho kết quả tương đối giống nhau. KNN là một thuật toán rất chậm so với các thuật toán khác, mất nhiều thời gian để huấn luyện. KNN không tốt về độ chuẩn xác (precision) và điểm F1. Nhìn chung, KNN không tốt cho tập dữ liệu này. Hình 13 (theo bài báo là Hình 14) cho thấy độ chính xác của KNN trên tập kiểm tra và vẽ biểu đồ heatmap.
    *(Hình 13 (heatmap - Hình 14 theo bài báo): HeatMap trên KNN avg word2vec. Cho thấy ma trận nhầm lẫn)*
    *(Hình 14 (đường cong - Hình 15 theo bài báo): Độ chính xác cho các giá trị k khác nhau trên uni_gram. Độ chính xác cao nhất quanh k=40-50)*

**(Trang 5)**

*   **Random Forest:** Các bước sau đã được tuân theo và kết luận:
    1.  Tiền xử lý văn bản.
    2.  Phân chia dựa trên thời gian toàn bộ tập dữ liệu thành train_data và test_data.
    3.  Huấn luyện bộ vector hóa trên train_data và sau đó áp dụng cùng bộ vector hóa trên cả train_data và test_data để biến đổi chúng thành các vector.
    4.  Sử dụng Random Forest làm công cụ ước tính trong GridSearchCV để tìm giá trị tối ưu của base_learners (n_estimators).
    5.  Sau khi có giá trị tối ưu của base_learners, huấn luyện lại Random Forest với giá trị tối ưu này và đưa ra dự đoán trên test_data.
    6.  Vẽ biểu đồ Cross_Validation Error VS Base_Learners(n_estimators).
    7.  Đánh giá: Accuracy, F1-Score, Precision, Recall.
    8.  Vẽ Heatmap Seaborn cho Confusion Matrix.
*   *(Phần mô tả về GBDT có vẻ bị lẫn với Random Forest ở bước 9-13)*
    9.  Sử dụng GBDT làm công cụ ước tính trong GridSearchCV để tìm giá trị tối ưu của base_learners, depth và learning_rate.
    10. Sau khi có giá trị tối ưu (của base_learners, depth và learning_rate), huấn luyện lại GBDT với các giá trị tối ưu này và đưa ra dự đoán trên test_data.
    11. Vẽ biểu đồ Cross-Validation Error vs bộ ba (Learning_rate, Max_depth, N_estimators).
    12. Đánh giá: Accuracy, F1-Score, Precision, Recall.
    13. Vẽ Heatmap Seaborn cho Confusion Matrix.
    Lặp lại từ Bước 3 đến Bước 13 cho mỗi trong bốn bộ vector hóa: Bag Of Words(BoW)(Hình 14), TFIDF(Hình 15), Avg Word2Vec(Hình 4.8 - không có hình này) và TFIDF Word2Vec(Hình 15).
    *(Hình 15: Lỗi Cross validation Error vs (Learning Rate, Max depth, N_estimators). Cho thấy lỗi thay đổi khi các siêu tham số của GBDT thay đổi)*

*   **Bảng 1 KNN:** Tóm tắt kết quả KNN với các vector hóa khác nhau. Độ chính xác cao nhất ~85.1% với Avg Word2Vec, nhưng F1-score thấp (0.6-0.7).
*   **Bảng 2 Logistic Regression:** Độ chính xác cao nhất ~93.7% với Bi-Gram BoW. Các phương pháp khác cũng cho độ chính xác cao (>91%).
*   **Bảng 3 SVM:** Độ chính xác cao nhất ~91.6% với TF-IDF. Bi-gram BoW cũng cho kết quả tốt (~90.7%). F1-score quanh 0.7.
*   **Bảng 4 Decision Tree:** Độ chính xác thấp hơn đáng kể, cao nhất ~85.1% với Bi-Gram BoW. F1-score thấp (cao nhất 0.529).
*   **Bảng 5 Naïve Bayes:** Độ chính xác cao nhất ~90.19% với MultinomialNB cho BoW (Unigram). F1-score ~0.7.
*   **Bảng 6 Naïve Bayes (chi tiết CV):** Xác nhận kết quả tương tự Bảng 5.
*   **Bảng 7 LSTM:** Mô hình học sâu RNN với LSTM cho độ chính xác cao nhất.
    *   1 lớp LSTM: ~91.93%
    *   2 lớp LSTM: ~91.97%
    *   3 lớp LSTM: ~92.17%
    *   4 lớp LSTM: ~92.18%

**(Trang 6 & 7)**

**V. KẾT LUẬN**

Sử dụng phương pháp học máy có giám sát giúp thu được kết quả. Hồi quy Logistic, Naïve Bayes cho độ chính xác tốt nhất trong các mô hình học máy. Trong khi đó, LSTM đã thắng trong cuộc đua vì nó là một mô hình học sâu và cho độ chính xác 92.1%. Có những tình cảm thuộc đủ loại nhưng phần lớn áp đảo có tình cảm tích cực. Kết quả của chúng tôi được xác minh thêm bởi độ chính xác đáng kinh ngạc của bộ phân loại hồi quy Logistic.



---

**Phân tích Bài báo**



**2. Phương pháp luận:**
*   **Dữ liệu:** Sử dụng bộ dữ liệu công khai và phổ biến "Amazon Fine Food Reviews" từ Kaggle, chứa khoảng 568.000 đánh giá. Dữ liệu được chia thành tập huấn luyện và kiểm tra dựa trên thời gian.
*   **Tiền xử lý:** Áp dụng các bước tiền xử lý văn bản tiêu chuẩn bao gồm loại bỏ dữ liệu trùng lặp, loại bỏ ký tự đặc biệt, số, dấu câu, từ dừng, thực hiện token hóa và stemming. Đây là những bước quan trọng để chuẩn hóa dữ liệu đầu vào cho các mô hình.
*   **Trích xuất đặc trưng/Vector hóa:** So sánh bốn kỹ thuật chính:
    *   Bag of Words (BoW): Unigram và Bigram.
    *   TF-IDF (Term Frequency-Inverse Document Frequency).
    *   Word2Vec (Trung bình và TF-IDF trọng số).
*   **Mô hình hóa:** Thử nghiệm một loạt các thuật toán học máy cổ điển và một mô hình học sâu:
    *   Học máy: Logistic Regression, Naïve Bayes (Bernoulli và Multinomial), K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Support Vector Machine (SVM).
    *   Học sâu: Long Short-Term Memory (LSTM), một dạng của Mạng Nơ-ron Hồi quy (RNN).
*   **Đánh giá:** Sử dụng các chỉ số phổ biến là Độ chính xác (Accuracy) và F1-Score. Sử dụng kỹ thuật Cross-Validation (GridSearchCV, RandomizedSearchCV) để tìm siêu tham số tối ưu.
*   **Trực quan hóa:** Sử dụng Word Clouds để khám phá các từ phổ biến, t-SNE để trực quan hóa không gian đặc trưng, biểu đồ phân phối điểm số, đường cong AUC để đánh giá mô hình, biểu đồ tầm quan trọng đặc trưng và ma trận nhầm lẫn (heatmap).

**3. Kết quả chính:**
*   **Hiệu suất mô hình:**
    *   Các mô hình học máy như Logistic Regression và Naïve Bayes (đặc biệt là MultinomialNB với BoW) đạt được độ chính xác cao (khoảng 90-93% cho LR, 88-90% cho NB) và F1-score khá tốt (khoảng 0.7). Điều này cho thấy chúng là những lựa chọn hiệu quả và mạnh mẽ cho bài toán này.
    *   SVM cũng cho kết quả tốt (khoảng 91.6% với TF-IDF).
    *   KNN và Decision Tree cho kết quả kém hơn đáng kể về cả độ chính xác và/hoặc F1-score. KNN còn bị đánh giá là chậm.
    *   Mô hình LSTM (học sâu) đạt được độ chính xác cao nhất, lên tới 92.18% với 4 lớp LSTM, vượt trội hơn một chút so với các mô hình học máy tốt nhất.
*   **Vector hóa:** Bi-gram BoW và TF-IDF thường mang lại kết quả tốt khi kết hợp với các bộ phân loại mạnh như Logistic Regression, SVM, Naive Bayes. Avg Word2Vec hoạt động tốt nhất với KNN nhưng tổng thể KNN không hiệu quả.
*   **Phân tích dữ liệu:** Word clouds và biểu đồ phân phối cho thấy sự phổ biến của các từ liên quan đến hương vị, chất lượng, mua hàng và có sự lệch rõ rệt về phía các đánh giá tích cực (điểm 4 và 5). Biểu đồ t-SNE cho thấy có sự phân cụm nhất định nhưng cũng có sự chồng chéo đáng kể giữa các lớp tình cảm, đặc biệt khi dùng Word2Vec, gợi ý rằng việc phân loại không hoàn toàn dễ dàng chỉ dựa trên các đặc trưng ngữ nghĩa đơn giản.
*   **Tầm quan trọng đặc trưng:** Các biểu đồ cho thấy các từ và cặp từ (bi-grams) cụ thể có vai trò quan trọng trong việc phân loại.

**4. Kết luận:**
*   Bài báo kết luận rằng cả các phương pháp học máy truyền thống (Logistic Regression, Naïve Bayes) và học sâu (LSTM) đều có thể đạt được hiệu suất cao trong việc phân tích tình cảm trên bộ dữ liệu này.
*   Logistic Regression được nhấn mạnh là một bộ phân loại hiệu quả, mang lại độ chính xác cao.
*   LSTM, là một mô hình học sâu, cho thấy khả năng đạt được độ chính xác cao nhất, mặc dù sự cải thiện so với Logistic Regression tốt nhất không quá lớn (khoảng 92.18% so với 93.7% - *Lưu ý: Có sự mâu thuẫn nhỏ giữa kết luận ca ngợi LR và bảng kết quả cho thấy LR có Acc cao nhất*). Tuy nhiên, kết luận cuối cùng lại nhấn mạnh LR. Có thể F1-score hoặc yếu tố khác đã ảnh hưởng đến nhận định cuối cùng này, hoặc có sự không nhất quán trong bài báo.

**5. Điểm mạnh:**
*   So sánh toàn diện nhiều kỹ thuật vector hóa và thuật toán phân loại trên cùng một bộ dữ liệu lớn và thực tế.
*   Sử dụng các phương pháp đánh giá và trực quan hóa tiêu chuẩn, giúp dễ dàng hiểu và so sánh kết quả.
*   Cung cấp một quy trình làm việc chuẩn cho bài toán phân tích tình cảm văn bản.

**6. Điểm yếu/Hạn chế:**
*   Không đi sâu vào phân tích lỗi hoặc các trường hợp mà mô hình dự đoán sai.
*   Thảo luận về mô hình LSTM còn khá sơ sài, chủ yếu chỉ trình bày kết quả cuối cùng mà không phân tích sâu hơn về kiến trúc hay quá trình huấn luyện.
*   Mặc dù đề cập đến dữ liệu lệch trong phần giới thiệu, bài báo không mô tả rõ các kỹ thuật cụ thể đã áp dụng để xử lý vấn đề này trong phần phương pháp hay kết quả.
*   Có sự không nhất quán nhỏ trong việc báo cáo và kết luận về mô hình nào là "tốt nhất" (LSTM có Acc cao nhất trong bảng LSTM, nhưng LR có Acc cao nhất trong bảng LR và được nhấn mạnh trong kết luận).


