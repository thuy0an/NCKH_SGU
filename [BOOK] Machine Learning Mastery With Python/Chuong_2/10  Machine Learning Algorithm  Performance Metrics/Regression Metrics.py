# 10.3.1 Mean Absolute Error 
# Cross Validation Regression MAE 
from pandas import read_csv 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LinearRegression 
filename = 'housing.csv' 
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] 
dataframe = read_csv(filename, delim_whitespace=True, names=names) 
array = dataframe.values 
X = array[:,0:13] 
Y = array[:,13] 
kfold = KFold(n_splits=10, random_state=7) 
model = LinearRegression() 
scoring = 'neg_mean_absolute_error' 
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) 
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())
# 10.3.2 Mean Squared Error 
# Cross Validation Regression MSE 
from pandas import read_csv 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LinearRegression 
filename = 'housing.csv' 
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] 
dataframe = read_csv(filename, delim_whitespace=True, names=names) 
array = dataframe.values 
X = array[:,0:13] 
Y = array[:,13] 
num_folds = 10 
kfold = KFold(n_splits=10, random_state=7) 
model = LinearRegression() 
scoring = 'neg_mean_squared_error' 
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) 
print("MSE: %.3f (%.3f)") % (results.mean(), results.std())
# 10.3.3 R2 Metric
# Cross Validation Regression R^2 
from pandas import read_csv 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LinearRegression 
filename = 'housing.csv' 
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] 
dataframe = read_csv(filename, delim_whitespace=True, names=names) 
array = dataframe.values 
X = array[:,0:13] 
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7) 
model = LinearRegression() 
scoring = 'r2' 
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) 
print("R^2: %.3f (%.3f)") % (results.mean(), results.std()) 
