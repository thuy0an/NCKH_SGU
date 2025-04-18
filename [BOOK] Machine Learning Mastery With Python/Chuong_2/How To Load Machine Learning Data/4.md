# 📘 Chương 4: Cách Tải Dữ Liệu Machine Learning
# Trong chương này, bạn sẽ học cách tải dữ liệu CSV, một trong những định dạng dữ liệu phổ biến nhất cho Machine Learning.
# Dữ liệu CSV có thể được tải bằng nhiều cách khác nhau trong Python, bao gồm:
# 1. Tải CSV với Thư viện chuẩn của Python
# 2. Tải CSV với NumPy
# 3. Tải CSV với Pandas

# 4.1 Cân Nhắc Khi Tải Dữ Liệu CSV
# Trước khi tải dữ liệu CSV, bạn cần lưu ý một số yếu tố quan trọng để tránh lỗi trong quá trình đọc dữ liệu.

# 4.1.1 Tiêu Đề Tệp
# - Tệp CSV có thể có tiêu đề, giúp tự động gán tên cho các cột.
# - Nếu không có tiêu đề, bạn cần tự gán tên cho các thuộc tính. Bạn phải chỉ rõ trong khi tải dữ liệu xem có tiêu đề hay không.

# 4.1.2 Bình Luận (Comments)
# - Các bình luận trong tệp CSV được đánh dấu bằng dấu # ở đầu dòng.
# - Bạn cần chỉ rõ xem dữ liệu của bạn có chứa bình luận hay không, để phương pháp tải dữ liệu có thể xử lý chính xác.

# 4.1.3 Dấu Phân Cách (Delimiter)
# - Dữ liệu CSV thường sử dụng dấu phẩy (',') để phân tách các trường. Tuy nhiên, nếu sử dụng dấu phân cách khác (ví dụ: tab hoặc dấu cách), bạn cần chỉ rõ trong khi tải dữ liệu.

# 4.1.4 Dấu Nháy (Quotes)
# - Các giá trị trong CSV có thể chứa khoảng trắng và thường được bao quanh bằng dấu nháy (quotation marks).
# - Python mặc định sử dụng dấu nháy kép (") làm ký tự bao quanh, nhưng bạn có thể thay đổi ký tự này nếu cần.

# 4.2 Dữ Liệu Pima Indians
# - Bộ dữ liệu Pima Indians chứa thông tin về bệnh tiểu đường của người Pima và có thể sử dụng để phân loại (classification).
# - Dữ liệu này có các thuộc tính số học và biến đầu ra là nhị phân (0 hoặc 1), rất thích hợp cho các bài học máy học.

# 4.3 Tải CSV với Thư Viện Chuẩn của Python
# - Python cung cấp thư viện CSV và hàm reader() để tải tệp CSV. Sau khi tải, bạn có thể chuyển đổi dữ liệu CSV thành mảng NumPy để sử dụng cho Machine Learning.
# Ví dụ:
import csv
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rb')  # Mở tệp CSV
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)  # Đọc tệp CSV
x = list(reader)  # Chuyển đổi tệp CSV thành danh sách
data = numpy.array(x).astype('float')  # Chuyển danh sách thành mảng NumPy và thay đổi kiểu dữ liệu
print(data.shape)  # In kích thước mảng

# 4.4 Tải CSV với NumPy
# - NumPy cung cấp hàm numpy.loadtxt() để tải dữ liệu CSV. Hàm này giả định rằng không có tiêu đề và tất cả dữ liệu đều có định dạng giống nhau.
# Ví dụ:
from numpy import loadtxt
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rb')
data = loadtxt(raw_data, delimiter=",")  # Tải dữ liệu và sử dụng dấu phẩy làm phân cách
print(data.shape)

# Tải CSV từ URL bằng NumPy:
from numpy import loadtxt
from urllib import urlopen
url = 'https://goo.gl/vhm1eU'  # URL chứa dữ liệu CSV
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=",")  # Tải dữ liệu từ URL
print(dataset.shape)

# 4.5 Tải CSV với Pandas
# - Pandas cung cấp hàm pandas.read_csv(), rất linh hoạt và dễ sử dụng. Đây là phương pháp được khuyến khích trong thực tế vì nó trả về một pandas.DataFrame, mà bạn có thể sử dụng ngay để tóm tắt và vẽ đồ thị.
# Ví dụ:
from pandas import read_csv
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']  # Tên các cột
data = read_csv(filename, names=names)  # Đọc tệp CSV và gán tên cột
print(data.shape)

# Tải CSV từ URL bằng Pandas:
from pandas import read_csv
url = 'https://goo.gl/vhm1eU'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']  # Tên các cột
data = read_csv(url, names=names)  # Tải dữ liệu từ URL và gán tên cột
print(data.shape)

# 4.6 Tổng Kết
# - Bạn đã học được ba phương pháp chính để tải dữ liệu CSV vào Python: sử dụng Thư viện Chuẩn, NumPy và Pandas.
# - Trong thực tế, tôi khuyến nghị bạn nên sử dụng Pandas vì nó cung cấp nhiều tính năng linh hoạt và mạnh mẽ hơn để xử lý dữ liệu.

# 4.6.1 Tiếp Theo
# - Sau khi tải dữ liệu, bước tiếp theo là khám phá và hiểu dữ liệu của bạn thông qua các thống kê mô tả đơn giản.
