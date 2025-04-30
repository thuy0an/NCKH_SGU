import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Hàm tải dữ liệu Boston House Price
def load_data():
    # Đọc file housing.csv, sử dụng dấu cách làm phân tách và đặt tên cột
    filename = 'housing.csv'
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataframe = pd.read_csv(filename, delim_whitespace=True, names=names)
    array = dataframe.values
    # Tách đặc trưng (X: 13 cột đầu) và mục tiêu (Y: cột MEDV)
    X = array[:, 0:13]
    Y = array[:, 13]
    return X, Y

# Hàm chạy thuật toán Hồi quy Tuyến tính
def linear_regression(X, Y):
    # Thiết lập kiểm định chéo 10-fold với seed cố định
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model = LinearRegression()
    scoring = 'neg_mean_squared_error'
    # Tính điểm kiểm định chéo
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    return results.mean()

# Hàm chạy thuật toán Hồi quy Ridge
def ridge_regression(X, Y):
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model = Ridge()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    return results.mean()

# Hàm chạy thuật toán Hồi quy LASSO
def lasso_regression(X, Y):
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model = Lasso()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    return results.mean()

# Hàm chạy thuật toán Hồi quy Elastic Net
def elastic_net_regression(X, Y):
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model = ElasticNet()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    return results.mean()

# Hàm chạy thuật toán k-Láng giềng gần nhất (KNN)
def knn_regression(X, Y):
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model = KNeighborsRegressor()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    return results.mean()

# Hàm chạy thuật toán Cây Quyết định (CART)
def cart_regression(X, Y):
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model = DecisionTreeRegressor()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    return results.mean()

# Hàm chạy thuật toán Hồi quy Máy Vector Hỗ trợ (SVR)
def svr_regression(X, Y):
，直    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model = SVR()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    return results.mean()

# Hàm hiển thị mục lục thuật toán
def display_menu():
    print("\nMục lục các thuật toán hồi quy:")
    print("1. Hồi quy Tuyến tính")
    print("2. Hồi quy Ridge")
    print("3. Hồi quy LASSO")
    print("4. Hồi quy Elastic Net")
    print("5. k-Láng giềng gần nhất")
    print("6. Cây Quyết định")
    print("7. Hồi quy Máy Vector Hỗ trợ")
    print("0. Thoát")
    print("Nhập số của thuật toán muốn chạy (0 để thoát): ")

# Hàm chính điều khiển hệ thống tương tác
def main():
    # Tải dữ liệu một lần
    try:
        X, Y = load_data()
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file housing.csv.")
        return

    # Ánh xạ số với tên thuật toán và hàm tương ứng
    algorithms = {
        1: ("Hồi quy Tuyến tính", linear_regression),
        2: ("Hồi quy Ridge", ridge_regression),
        3: ("Hồi quy LASSO", lasso_regression),
        4: ("Hồi quy Elastic Net", elastic_net_regression),
        5: ("k-Láng giềng gần nhất", knn_regression),
        6: ("Cây Quyết định", cart_regression),
        7: ("Hồi quy Máy Vector Hỗ trợ", svr_regression)
    }

    while True:
        display_menu()
        try:
            choice = int(input())
            if choice == 0:
                print("Thoát chương trình.")
                break
            if choice in algorithms:
                algo_name, algo_func = algorithms[choice]
                print(f"\nĐang chạy {algo_name}...")
                result = algo_func(X, Y)
                print(f"Lỗi bình phương trung bình: {result:.6f}")
            else:
                print("Lựa chọn không hợp lệ. Vui lòng chọn số từ 0 đến 7.")
        except ValueError:
            print("Đầu vào không hợp lệ. Vui lòng nhập một số.")

if __name__ == "__main__":
    main()
