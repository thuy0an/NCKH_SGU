from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer

# Load dataset
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values

# Separate input and output
X = array[:,0:8]
Y = array[:,8]

# ------------------------
# 1. Rescale Data (0 to 1)
# ------------------------
scaler_minmax = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler_minmax.fit_transform(X)
set_printoptions(precision=3)
print("Rescaled Data (MinMaxScaler):")
print(rescaledX[0:5,:])

# ------------------------
# 2. Standardize Data (0 mean, 1 stdev)
# ------------------------
scaler_std = StandardScaler().fit(X)
standardizedX = scaler_std.transform(X)
print("\nStandardized Data (StandardScaler):")
print(standardizedX[0:5,:])

# ------------------------
# 3. Normalize Data (length of 1)
# ------------------------
scaler_norm = Normalizer().fit(X)
normalizedX = scaler_norm.transform(X)
print("\nNormalized Data (Normalizer):")
print(normalizedX[0:5,:])

# ------------------------
# 4. Binarize Data (threshold=0.0)
# ------------------------
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
print("\nBinarized Data (Binarizer):")
print(binaryX[0:5,:])
