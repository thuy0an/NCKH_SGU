# Chương 11 - Kiểm tra  các thuật toán phân loại

Kiểm tra (Spot-checking) là một cách để khám phá những thuật toán nào hoạt động tốt trên bài toán học máy của bạn. Bạn không thể biết trước thuật toán nào phù hợp nhất với bài toán của mình. Bạn phải thử nghiệm một số phương pháp và tập trung vào những phương pháp chứng tỏ chúng hứa hẹn nhất. Trong chương này, bạn sẽ khám phá sáu thuật toán học máy mà bạn có thể sử dụng khi kiểm tra  bài toán phân loại của mình trong Python với scikit-learn. Sau khi hoàn thành bài học này, bạn sẽ biết:


## 11.1 Kiểm tra  thuật toán

Không thể biết trước thuật toán nào sẽ hoạt động tốt nhất trên tập dữ liệu của mình. Bạn phải sử dụng phương pháp thử nghiệm để khám phá một danh sách ngắn các thuật toán hoạt động tốt trên bài toán của bạn, sau đó bạn có thể tập trung vào chúng và tinh chỉnh thêm. Tôi gọi quá trình này là kiểm tra  (spot-checking).

Câu hỏi không phải là: Tôi nên sử dụng thuật toán nào cho tập dữ liệu của mình? Thay vào đó là: Tôi nên kiểm tra  những thuật toán nào trên tập dữ liệu của mình? Bạn có thể đoán thuật toán nào có thể hoạt động tốt trên tập dữ liệu của bạn, và đây có thể là một điểm khởi đầu tốt. Tôi khuyên bạn nên thử kết hợp nhiều thuật toán và xem thuật toán nào giỏi trong việc phát hiện cấu trúc trong dữ liệu của bạn. Dưới đây là một số gợi ý khi kiểm tra  các thuật toán trên tập dữ liệu của bạn:

- Thử kết hợp các kiểu biểu diễn thuật toán khác nhau (ví dụ: instance-based và tree-based).
- Thử kết hợp các thuật toán học tập khác nhau (ví dụ: các thuật toán khác nhau để học cùng một loại biểu diễn).
- Thử kết hợp các loại mô hình khác nhau (ví dụ: hàm tuyến tính và phi tuyến tính hoặc tham số và phi tham số).

Hãy cụ thể hơn. Trong phần tiếp theo, chúng ta sẽ xem xét các thuật toán mà bạn có thể sử dụng để kiểm tra  trong dự án học máy phân loại tiếp theo của bạn trong Python.

## 11.2 Tổng quan về các thuật toán

Chúng ta sẽ xem xét sáu thuật toán phân loại mà bạn có thể kiểm tra  trên tập dữ liệu của mình. Bắt đầu với hai thuật toán học máy tuyến tính:

- Hồi quy Logistic (Logistic Regression).
- Phân tích phân biệt tuyến tính (Linear Discriminant Analysis).

Sau đó xem xét bốn thuật toán học máy phi tuyến tính:

- k-Nearest Neighbors (k-láng giềng gần nhất).
- Naive Bayes (Bayes ngây thơ).
- Cây phân loại và hồi quy (Classification and Regression Trees - CART).
- Máy vector hỗ trợ (Support Vector Machines - SVM).

Mỗi thuật toán được minh họa trên tập dữ liệu Pima Indians Onset of Diabetes. Một quy trình kiểm tra sử dụng kiểm tra chéo 10-fold được sử dụng để minh họa cách kiểm tra  từng thuật toán học máy và các phép đo độ chính xác trung bình được sử dụng để chỉ ra hiệu suất thuật toán.

## 11.3 Thuật toán học máy tuyến tính

Phần này minh họa các công thức tối thiểu để sử dụng hai thuật toán học máy tuyến tính: hồi quy logistic và phân tích phân biệt tuyến tính.

### 11.3.1 Hồi quy Logistic

Hồi quy logistic giả định phân phối Gaussian cho các biến đầu vào số học và có thể mô hình hóa các bài toán phân loại nhị phân. Bạn có thể xây dựng mô hình hồi quy logistic bằng lớp `LogisticRegression`.

```python
# Logistic Regression Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Chạy ví dụ in ra độ chính xác ước tính trung bình: 0.76951469583

### 11.3.2 Phân tích phân biệt tuyến tính

Phân tích phân biệt tuyến tính hay LDA là một kỹ thuật thống kê cho phân loại nhị phân và đa lớp. Nó cũng giả định phân phối Gaussian cho các biến đầu vào số học. Bạn có thể xây dựng mô hình LDA bằng lớp `LinearDiscriminantAnalysis`.

```python
# LDA Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Chạy ví dụ in ra độ chính xác ước tính trung bình: 0.773462064252

## 11.4 Thuật toán học máy phi tuyến tính

Phần này minh họa các công thức tối thiểu để sử dụng 4 thuật toán học máy phi tuyến tính.

### 11.4.1 k-Nearest Neighbors

Thuật toán k-Nearest Neighbors (hoặc KNN) sử dụng một phép đo khoảng cách để tìm k phần tử tương tự nhất trong dữ liệu huấn luyện cho một phần tử mới và lấy kết quả trung bình của các láng giềng làm dự đoán. Bạn có thể xây dựng mô hình KNN bằng lớp `KNeighborsClassifier`.

```python
# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Chạy ví dụ in ra độ chính xác ước tính trung bình: 0.726555023923

### 11.4.2 Naive Bayes

Naive Bayes tính toán xác suất của mỗi lớp và xác suất có điều kiện của mỗi lớp với từng giá trị đầu vào. Những xác suất này được ước tính cho dữ liệu mới và nhân với nhau, với giả định rằng chúng đều độc lập (một giả định đơn giản hoặc "ngây thơ"). Khi làm việc với dữ liệu giá trị thực, một phân phối Gaussian được giả định để dễ dàng ước tính xác suất cho các biến đầu vào bằng Hàm mật độ xác suất Gaussian. Bạn có thể xây dựng mô hình Naive Bayes bằng lớp `GaussianNB`.

```python
# Gaussian Naive Bayes Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Chạy ví dụ in ra độ chính xác ước tính trung bình: 0.75517771702

### 11.4.3 Cây phân loại và hồi quy

Cây phân loại và hồi quy (CART hoặc chỉ là cây quyết định) xây dựng một cây nhị phân từ dữ liệu huấn luyện. Các điểm chia được chọn một cách tham lam bằng cách đánh giá từng thuộc tính và từng giá trị của từng thuộc tính trong dữ liệu huấn luyện để giảm thiểu hàm chi phí (như chỉ số Gini). Bạn có thể xây dựng mô hình CART bằng lớp `DecisionTreeClassifier`.

```python
# CART Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Chạy ví dụ in ra độ chính xác ước tính trung bình: 0.692600820232

### 11.4.4 Máy vector hỗ trợ

Máy vector hỗ trợ (Support Vector Machines hay SVM) tìm kiếm đường thẳng phân tách tốt nhất giữa hai lớp. Những phần tử dữ liệu gần nhất với đường thẳng phân tách tốt nhất được gọi là các vector hỗ trợ và ảnh hưởng đến vị trí đặt đường thẳng. SVM đã được mở rộng để hỗ trợ nhiều lớp. Đặc biệt quan trọng là việc sử dụng các hàm kernel khác nhau thông qua tham số kernel. Một Radial Basis Function mạnh mẽ được sử dụng theo mặc định. Bạn có thể xây dựng mô hình SVM bằng lớp `SVC`.

```python
# SVM Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

Chạy ví dụ in ra độ chính xác ước tính trung bình: 0.651025290499

## 11.5 Tóm tắt

Trong chương này, bạn đã khám phá 6 thuật toán học máy mà bạn có thể sử dụng để kiểm tra  trên bài toán phân loại của mình trong Python sử dụng scikit-learn. Cụ thể, bạn đã học cách kiểm tra  hai thuật toán học máy tuyến tính: Hồi quy Logistic và Phân tích phân biệt tuyến tính. Bạn cũng đã học cách kiểm tra  bốn thuật toán phi tuyến tính: k-Nearest Neighbors, Naive Bayes, Cây phân loại và hồi quy, và Máy vector hỗ trợ.
