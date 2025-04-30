# Lưu và tải mô hình học máy
# Script này triển khai hai phương pháp lưu và tải mô hình từ Chương 17:
# 1. Sử dụng Pickle để lưu và tải mô hình
# 2. Sử dụng Joblib để lưu và tải mô hình
# Người dùng chọn phương pháp để chạy qua giao diện dòng lệnh.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import joblib

# Hàm tải và chia dữ liệu Pima Indians Diabetes
def load_and_split_data():
    """
    Tải tập dữ liệu Pima Indians Diabetes từ file CSV.
    Chia dữ liệu thành tập huấn luyện (67%) và tập kiểm tra (33%).
    Trả về X_train, X_test, Y_train, Y_test.
    """
    filename = 'pima-indians-diabetes.data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(filename, names=names)
    array = dataframe.values
    X = array[:, 0:8]  # Đặc trưng: 8 cột đầu
    Y = array[:, 8]    # Nhãn: cột cuối (class)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
    return X_train, X_test, Y_train, Y_test

# Phương thức 1: Lưu và tải mô hình với Pickle
def run_pickle_model(X_train, X_test, Y_train, Y_test):
    """
    Huấn luyện mô hình Logistic Regression, lưu bằng Pickle, sau đó tải và đánh giá trên tập kiểm tra.
    In độ chính xác của mô hình.
    """
    print("\nĐang chạy lưu và tải mô hình với Pickle...")
    # Huấn luyện mô hình
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    
    # Lưu mô hình vào file
    filename = 'finalized_model_pickle.sav'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    # Tải mô hình từ file
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    
    # Đánh giá mô hình trên tập kiểm tra
    result = loaded_model.score(X_test, Y_test)
    print(f"Độ chính xác trên tập kiểm tra: {result:.3f}")

# Phương thức 2: Lưu và tải mô hình với Joblib
def run_joblib_model(X_train, X_test, Y_train, Y_test):
    """
    Huấn luyện mô hình Logistic Regression, lưu bằng Joblib, sau đó tải và đánh giá trên tập kiểm tra.
    In độ chính xác của mô hình.
    """
    print("\nĐang chạy lưu và tải mô hình với Joblib...")
    # Huấn luyện mô hình
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    
    # Lưu mô hình vào file
    filename = 'finalized_model_joblib.sav'
    joblib.dump(model, filename)
    
    # Tải mô hình từ file
    loaded_model = joblib.load(filename)
    
    # Đánh giá mô hình trên tập kiểm tra
    result = loaded_model.score(X_test, Y_test)
    print(f"Độ chính xác trên tập kiểm tra: {result:.3f}")

# Hàm chính để tương tác với người dùng
def main():
    """
    Hàm chính: Tải và chia dữ liệu, hiển thị menu, và xử lý lựa chọn của người dùng.
    """
    # Tải và chia dữ liệu
    try:
        X_train, X_test, Y_train, Y_test = load_and_split_data()
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'pima-indians-diabetes.data.csv'. Vui lòng đảm bảo file nằm cùng thư mục.")
        return
    
    while True:
        # Hiển thị menu
        print("\n=== Menu Lưu và Tải Mô Hình Học Máy ===")
        print("1. Lưu và tải mô hình với Pickle")
        print("2. Lưu và tải mô hình với Joblib")
        print("0. Thoát")
        
        # Nhận input từ người dùng
        choice = input("Nhập lựa chọn của bạn (0-2): ")
        
        # Xử lý lựa chọn
        if choice == '1':
            run_pickle_model(X_train, X_test, Y_train, Y_test)
        elif choice == '2':
            run_joblib_model(X_train, X_test, Y_train, Y_test)
        elif choice == '0':
            print("Thoát chương trình. Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng nhập 0, 1, hoặc 2.")

# Chạy chương trình
if __name__ == "__main__":
    main()
