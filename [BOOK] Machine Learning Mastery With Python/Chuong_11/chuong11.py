
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Hàm tải dữ liệu
def load_data():
    filename = 'pima-indians-diabetes.data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(filename, names=names)
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]
    return X, Y

# Hàm thiết lập KFold
def get_kfold():
    return KFold(n_splits=10, random_state=7, shuffle=True)

# Hàm chạy Logistic Regression
def run_logistic_regression(X, Y):
    kfold = get_kfold()
    model = LogisticRegression()
    results = cross_val_score(model, X, Y, cv=kfold)
    return results.mean()

# Hàm chạy Linear Discriminant Analysis
def run_lda(X, Y):
    kfold = get_kfold()
    model = LinearDiscriminantAnalysis()
    results = cross_val_score(model, X, Y, cv=kfold)
    return results.mean()

# Hàm chạy k-Nearest Neighbors
def run_knn(X, Y):
    kfold = get_kfold()
    model = KNeighborsClassifier()
    results = cross_val_score(model, X, Y, cv=kfold)
    return results.mean()

# Hàm chạy Naive Bayes
def run_naive_bayes(X, Y):
    kfold = get_kfold()
    model = GaussianNB()
    results = cross_val_score(model, X, Y, cv=kfold)
    return results.mean()

# Hàm chạy Classification and Regression Trees
def run_cart(X, Y):
    kfold = get_kfold()
    model = DecisionTreeClassifier()
    results = cross_val_score(model, X, Y, cv=kfold)
    return results.mean()

# Hàm chạy Support Vector Machines
def run_svm(X, Y):
    kfold = get_kfold()
    model = SVC()
    results = cross_val_score(model, X, Y, cv=kfold)
    return results.mean()

# Hàm hiển thị menu
def display_menu():
    print("\n=== Mục lục thuật toán phân loại ===")
    print("1. Logistic Regression")
    print("2. Linear Discriminant Analysis (LDA)")
    print("3. k-Nearest Neighbors (KNN)")
    print("4. Naive Bayes")
    print("5. Classification and Regression Trees (CART)")
    print("6. Support Vector Machines (SVM)")
    print("Nhập số từ 1-6 để chọn thuật toán hoặc bất kỳ ký tự khác để thoát.")

# Hàm chính
def main():
    # Tải dữ liệu
    try:
        X, Y = load_data()
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy tệp 'pima-indians-diabetes.data.csv'. Vui lòng đảm bảo tệp tồn tại trong thư mục hiện tại.")
        return

    while True:
        # Hiển thị menu
        display_menu()
        
        # Nhập lựa chọn từ người dùng
        choice = input("Lựa chọn của bạn: ")

        # Dictionary ánh xạ lựa chọn với hàm thuật toán
        algorithms = {
            '1': ('Logistic Regression', run_logistic_regression),
            '2': ('Linear Discriminant Analysis', run_lda),
            '3': ('k-Nearest Neighbors', run_knn),
            '4': ('Naive Bayes', run_naive_bayes),
            '5': ('Classification and Regression Trees', run_cart),
            '6': ('Support Vector Machines', run_svm)
        }

        # Kiểm tra lựa chọn
        if choice in algorithms:
            algo_name, algo_func = algorithms[choice]
            print(f"\nĐang chạy {algo_name}...")
            accuracy = algo_func(X, Y)
            print(f"Độ chính xác trung bình: {accuracy:.4f} ({accuracy*100:.2f}%)")
        else:
            print("Thoát chương trình.")
            break

        # Hỏi người dùng có muốn tiếp tục không
        continue_choice = input("\nBạn có muốn chạy thuật toán khác không? (y/n): ").lower()
        if continue_choice != 'y':
            print("Thoát chương trình.")
            break

if __name__ == "__main__":
    main()
