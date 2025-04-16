import pandas as pd

# Đọc dữ liệu
df = pd.read_csv('your_dataset.csv')

# 5.1 Peek at Your Data
print("First 5 rows of data:")
print(df.head())

# 5.2 Dimensions of Your Data
print("\nNumber of rows and columns:", df.shape)

# 5.3 Data Type For Each Attribute
print("\nData types of each attribute:")
print(df.dtypes)

# 5.4 Descriptive Statistics
print("\nDescriptive statistics:")
print(df.describe())

# 5.5 Class Distribution (Classification Only)
if 'Class' in df.columns:
    print("\nClass distribution:")
    print(df['Class'].value_counts())

# 5.6 Correlations Between Attributes
print("\nCorrelations between attributes:")
print(df.corr())

# 5.7 Skew of Univariate Distributions
print("\nSkew of univariate distributions:")
print(df.skew())
