# Tự động hóa quy trình học máy với Pipelines
# Script này tổ chức hai pipeline từ Chương 14 thành các hàm riêng:
# 1. Pipeline 1: Chuẩn hóa dữ liệu + LinearDiscriminantAnalysis
# 2. Pipeline 2: Trích xuất đặc trưng (PCA + SelectKBest) + LogisticRegression
# Người dùng chọn pipeline để chạy qua giao diện dòng lệnh.

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

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

# Pipeline 1: Chuẩn hóa dữ liệu + LinearDiscriminantAnalysis
def run_pipeline1(X, Y):
    """
    Chạy Pipeline 1: Chuẩn hóa dữ liệu với StandardScaler và huấn luyện mô hình LinearDiscriminantAnalysis.
    Đánh giá bằng 10-fold cross-validation và in kết quả.
    """
    print("\nĐang chạy Pipeline 1: Chuẩn hóa + LinearDiscriminantAnalysis")
    # Tạo pipeline
    estimators = []
    estimators.append(('standardize', StandardScaler()))  # Bước 1: Chuẩn hóa dữ liệu
    estimators.append(('lda', LinearDiscriminantAnalysis()))  # Bước 2: Mô hình LDA
    pipeline = Pipeline(estimators)
    
    # Đánh giá pipeline bằng 10-fold cross-validation
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='accuracy')
    print(f"Độ chính xác trung bình: {results.mean():.3f} (Độ lệch chuẩn: {results.std():.3f})")

# Pipeline 2: Trích xuất đặc trưng + LogisticRegression
def run_pipeline2(X, Y):
    """
    Chạy Pipeline 2: Trích xuất đặc trưng với FeatureUnion (PCA + SelectKBest) và huấn luyện LogisticRegression.
    Đánh giá bằng 10-fold cross-validation và in kết quả.
    """
    print("\nĐang chạy Pipeline 2: FeatureUnion(PCA + SelectKBest) + LogisticRegression")
    # Tạo FeatureUnion để trích xuất đặc trưng
    features = []
    features.append(('pca', PCA(n_components=3)))  # Bước 1: PCA với 3 thành phần
    features.append(('select_best', SelectKBest(k=6)))  # Bước 2: Chọn 6 đặc trưng tốt nhất
    feature_union = FeatureUnion(features)
    
    # Tạo pipeline
    estimators = []
    estimators.append(('feature_union', feature_union))  # Bước 1: Trích xuất đặc trưng
    estimators.append(('logistic', LogisticRegression(max_iter=1000)))  # Bước 2: Mô hình Logistic Regression
    pipeline = Pipeline(estimators)
    
    # Đánh giá pipeline bằng 10-fold cross-validation
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='accuracy')
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
        print("\n=== Menu Pipeline Học Máy ===")
        print("1. Chạy Pipeline 1 (Chuẩn hóa + LinearDiscriminantAnalysis)")
        print("2. Chạy Pipeline 2 (FeatureUnion + LogisticRegression)")
        print("0. Thoát")
        
        # Nhận input từ người dùng
        choice = input("Nhập lựa chọn của bạn (0-2): ")
        
        # Xử lý lựa chọn
        if choice == '1':
            run_pipeline1(X, Y)
        elif choice == '2':
            run_pipeline2(X, Y)
        elif choice == '0':
            print("Thoát chương trình. Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng nhập 0, 1, hoặc 2.")

# Chạy chương trình
if __name__ == "__main__":
    main()
