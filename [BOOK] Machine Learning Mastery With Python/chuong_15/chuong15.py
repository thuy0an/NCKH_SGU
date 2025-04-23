# Tự động hóa các thuật toán Ensemble trong Học máy
# Script này triển khai các thuật toán Ensemble từ Chương 15:
# 1. Bagged Decision Trees
# 2. Random Forest
# 3. Extra Trees
# 4. AdaBoost
# 5. Stochastic Gradient Boosting
# 6. Voting Ensemble
# Người dùng chọn thuật toán để chạy qua giao diện dòng lệnh.

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

# Phương thức 1: Bagged Decision Trees
def run_bagged_decision_trees(X, Y):
    """
    Chạy thuật toán Bagged Decision Trees với 100 cây quyết định.
    Đánh giá bằng 10-fold cross-validation và in kết quả.
    """
    print("\nĐang chạy Bagged Decision Trees...")
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cart = DecisionTreeClassifier()
    num_trees = 100
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(f"Độ chính xác trung bình: {results.mean():.3f} (Độ lệch chuẩn: {results.std():.3f})")

# Phương thức 2: Random Forest
def run_random_forest(X, Y):
    """
    Chạy thuật toán Random Forest với 100 cây và 3 đặc trưng ngẫu nhiên mỗi lần phân tách.
    Đánh giá bằng 10-fold cross-validation và in kết quả.
    """
    print("\nĐang chạy Random Forest...")
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    num_trees = 100
    max_features = 3
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, random_state=seed)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(f"Độ chính xác trung bình: {results.mean():.3f} (Độ lệch chuẩn: {results.std():.3f})")

# Phương thức 3: Extra Trees
def run_extra_trees(X, Y):
    """
    Chạy thuật toán Extra Trees với 100 cây và 7 đặc trưng ngẫu nhiên mỗi lần phân tách.
    Đánh giá bằng 10-fold cross-validation và in kết quả.
    """
    print("\nĐang chạy Extra Trees...")
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    num_trees = 100
    max_features = 7
    model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features, random_state=seed)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(f"Độ chính xác trung bình: {results.mean():.3f} (Độ lệch chuẩn: {results.std():.3f})")

# Phương thức 4: AdaBoost
def run_adaboost(X, Y):
    """
    Chạy thuật toán AdaBoost với 30 cây quyết định.
    Đánh giá bằng 10-fold cross-validation và in kết quả.
    """
    print("\nĐang chạy AdaBoost...")
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    num_trees = 30
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(f"Độ chính xác trung bình: {results.mean():.3f} (Độ lệch chuẩn: {results.std():.3f})")

# Phương thức 5: Stochastic Gradient Boosting
def run_gradient_boosting(X, Y):
    """
    Chạy thuật toán Stochastic Gradient Boosting với 100 cây.
    Đánh giá bằng 10-fold cross-validation và in kết quả.
    """
    print("\nĐang chạy Stochastic Gradient Boosting...")
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    num_trees = 100
    model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(f"Độ chính xác trung bình: {results.mean():.3f} (Độ lệch chuẩn: {results.std():.3f})")

# Phương thức 6: Voting Ensemble
def run_voting_ensemble(X, Y):
    """
    Chạy thuật toán Voting Ensemble kết hợp LogisticRegression, DecisionTreeClassifier, và SVC.
    Đánh giá bằng 10-fold cross-validation và in kết quả.
    """
    print("\nĐang chạy Voting Ensemble...")
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    # Tạo các mô hình con
    estimators = []
    model1 = LogisticRegression(max_iter=1000)
    estimators.append(('logistic', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = SVC()
    estimators.append(('svm', model3))
    # Tạo ensemble model
    ensemble = VotingClassifier(estimators)
    results = cross_val_score(ensemble, X, Y, cv=kfold)
    print(f"Độ chính xác trung bình: {results.mean():.3f} (Độ lệch chuẩn: {results.std():.3f})")

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
        print("\n=== Menu Các Thuật Toán Ensemble ===")
        print("1. Bagged Decision Trees")
        print("2. Random Forest")
        print("3. Extra Trees")
        print("4. AdaBoost")
        print("5. Stochastic Gradient Boosting")
        print("6. Voting Ensemble")
        print("0. Thoát")
        
        # Nhận input từ người dùng
        choice = input("Nhập lựa chọn của bạn (0-6): ")
        
        # Xử lý lựa chọn
        if choice == '1':
            run_bagged_decision_trees(X, Y)
        elif choice == '2':
            run_random_forest(X, Y)
        elif choice == '3':
            run_extra_trees(X, Y)
        elif choice == '4':
            run_adaboost(X, Y)
        elif choice == '5':
            run_gradient_boosting(X, Y)
        elif choice == '6':
            run_voting_ensemble(X, Y)
        elif choice == '0':
            print("Thoát chương trình. Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng nhập 0, 1, 2, 3, 4, 5, hoặc 6.")

# Chạy chương trình
if __name__ == "__main__":
    main()
