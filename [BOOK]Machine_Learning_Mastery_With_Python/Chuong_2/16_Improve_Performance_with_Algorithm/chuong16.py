# Tự động hóa tối ưu hóa tham số thuật toán trong Học máy
# Script này triển khai hai phương pháp tối ưu hóa tham số từ Chương 16:
# 1. Grid Search Parameter Tuning
# 2. Random Search Parameter Tuning
# Người dùng chọn phương pháp để chạy qua giao diện dòng lệnh.

import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Hàm tải dữ liệu Pima Indians Diabetes
def load_data():
    """
    Tải tập dữ liệu Pima Indians Diabetes từ file CSV.
    Trả về X (đặc trưng) và Y (nhãn).
    """
    filename = 'pima-indians-diabetes.data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(filename, names=names)
    array = dataframe.values
    X = array[:, 0:8]  # Đặc trưng: 8 cột đầu
    Y = array[:, 8]    # Nhãn: cột cuối (class)
    return X, Y

# Phương thức 1: Grid Search Parameter Tuning
def run_grid_search(X, Y):
    """
    Chạy Grid Search để tìm giá trị alpha tối ưu cho Ridge Regression.
    Thử các giá trị alpha: [1, 0.1, 0.01, 0.001, 0.0001, 0].
    In điểm số tốt nhất và giá trị alpha tương ứng.
    """
    print("\nĐang chạy Grid Search Parameter Tuning...")
    # Định nghĩa lưới tham số
    alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
    param_grid = dict(alpha=alphas)
    # Tạo mô hình Ridge
    model = Ridge()
    # Tạo GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    # Huấn luyện và tìm tham số tốt nhất
    grid.fit(X, Y)
    print(f"Điểm số tốt nhất: {grid.best_score_:.3f}")
    print(f"Giá trị alpha tốt nhất: {grid.best_estimator_.alpha}")

# Phương thức 2: Random Search Parameter Tuning
def run_random_search(X, Y):
    """
    Chạy Random Search để tìm giá trị alpha tối ưu cho Ridge Regression.
    Lấy mẫu ngẫu nhiên 100 giá trị alpha trong khoảng [0, 1].
    In điểm số tốt nhất và giá trị alpha tương ứng.
    """
    print("\nĐang chạy Random Search Parameter Tuning...")
    # Định nghĩa phân phối ngẫu nhiên cho alpha
    param_grid = {'alpha': uniform(loc=0, scale=1)}  # Phân phối đều từ 0 đến 1
    # Tạo mô hình Ridge
    model = Ridge()
    # Tạo RandomizedSearchCV
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, random_state=7)
    # Huấn luyện và tìm tham số tốt nhất
    rsearch.fit(X, Y)
    print(f"Điểm số tốt nhất: {rsearch.best_score_:.3f}")
    print(f"Giá trị alpha tốt nhất: {rsearch.best_estimator_.alpha:.6f}")

# Hàm chính để tương tác với người dùng
def main():
    """
    Hàm chính: Tải dữ liệu, hiển thị menu, và xử lý lựa chọn của người dùng.
    """
    # Tải dữ liệu
    try:
        X, Y = load_data()
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'pima-indians-diabetes.data.csv'. Vui lòng đảm bảo file nằm cùng thư mục.")
        return
    
    while True:
        # Hiển thị menu
        print("\n=== Menu Tối Ưu Hóa Tham Số Thuật Toán ===")
        print("1. Grid Search Parameter Tuning")
        print("2. Random Search Parameter Tuning")
        print("0. Thoát")
        
        # Nhận input từ người dùng
        choice = input("Nhập lựa chọn của bạn (0-2): ")
        
        # Xử lý lựa chọn
        if choice == '1':
            run_grid_search(X, Y)
        elif choice == '2':
            run_random_search(X, Y)
        elif choice == '0':
            print("Thoát chương trình. Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng nhập 0, 1, hoặc 2.")

# Chạy chương trình
if __name__ == "__main__":
    main()
