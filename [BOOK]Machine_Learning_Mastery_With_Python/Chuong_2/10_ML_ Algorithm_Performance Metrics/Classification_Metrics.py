# Cross Validation Classification Accuracy 
from pandas import read_csv 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LogisticRegression 
filename = 'pima-indians-diabetes.data.csv' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names) 
array = dataframe.values 
X = array[:,0:8] 
Y = array[:,8] 
kfold = KFold(n_splits=10, random_state=7) 
model = LogisticRegression() 
scoring = 'accuracy' 
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) 
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
# Cross Validation Classification LogLoss 
from pandas import read_csv 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LogisticRegression 
filename = 'pima-indians-diabetes.data.csv' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names) 
array = dataframe.values 
X = array[:,0:8] 
Y = array[:,8] 
kfold = KFold(n_splits=10, random_state=7) 
model = LogisticRegression() 
scoring = 'neg_log_loss' 
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) 
print("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
# Cross Validation Classification ROC AUC 
from pandas import read_csv 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LogisticRegression 
filename = 'pima-indians-diabetes.data.csv' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names) 
array = dataframe.values 
X = array[:,0:8] 
Y = array[:,8] 
kfold = KFold(n_splits=10, random_state=7) 
model = LogisticRegression() 
scoring = 'roc_auc' 
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) 
print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
# Confusion Matrix 
# Cross Validation Classification Confusion Matrix 
from pandas import read_csv 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
filename = 'pima-indians-diabetes.data.csv' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names) 
array = dataframe.values 
X = array[:,0:8] 
Y = array[:,8] 
test_size = 0.33
seed = 7 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed) 
model = LogisticRegression() 
model.fit(X_train, Y_train) 
predicted = model.predict(X_test) 
matrix = confusion_matrix(Y_test, predicted) 
print(matrix) 
# Cross Validation Classification Report 
from pandas import read_csv 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report 
filename = 'pima-indians-diabetes.data.csv' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names) 
array = dataframe.values 
X = array[:,0:8] 
Y = array[:,8] 
test_size = 0.33 
seed = 7 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed) 
model = LogisticRegression() 
model.fit(X_train, Y_train) 
predicted = model.predict(X_test) 
report = classification_report(Y_test, predicted) 
print(report)
