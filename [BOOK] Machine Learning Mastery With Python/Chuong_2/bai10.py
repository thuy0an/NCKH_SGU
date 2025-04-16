# Import necessary libraries
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report

# Dataset for classification
filename_classification = 'pima-indians-diabetes.data.csv'
names_classification = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe_classification = read_csv(filename_classification, names=names_classification)
array_classification = dataframe_classification.values
X_classification = array_classification[:, 0:8]
Y_classification = array_classification[:, 8]

# Dataset for regression
filename_regression = 'housing.csv'
names_regression = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe_regression = read_csv(filename_regression, delim_whitespace=True, names=names_regression)
array_regression = dataframe_regression.values
X_regression = array_regression[:, 0:13]
Y_regression = array_regression[:, 13]

# 10-fold Cross Validation
kfold = KFold(n_splits=10, random_state=7)

# ========================== Classification Metrics ==========================

# 1. Classification Accuracy
model_classification = LogisticRegression()
scoring_accuracy = 'accuracy'
results_accuracy = cross_val_score(model_classification, X_classification, Y_classification, cv=kfold, scoring=scoring_accuracy)
print(f"Classification Accuracy: {results_accuracy.mean():.3f} ({results_accuracy.std():.3f})")

# 2. Logarithmic Loss
scoring_logloss = 'neg_log_loss'
results_logloss = cross_val_score(model_classification, X_classification, Y_classification, cv=kfold, scoring=scoring_logloss)
print(f"Logloss: {results_logloss.mean():.3f} ({results_logloss.std():.3f})")

# 3. Area Under ROC Curve (AUC)
scoring_auc = 'roc_auc'
results_auc = cross_val_score(model_classification, X_classification, Y_classification, cv=kfold, scoring=scoring_auc)
print(f"AUC: {results_auc.mean():.3f} ({results_auc.std():.3f})")

# 4. Confusion Matrix
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X_classification, Y_classification, test_size=test_size, random_state=seed)
model_classification.fit(X_train, Y_train)
predicted_classification = model_classification.predict(X_test)
matrix_classification = confusion_matrix(Y_test, predicted_classification)
print(f"Confusion Matrix:\n{matrix_classification}")

# 5. Classification Report
report_classification = classification_report(Y_test, predicted_classification)
print(f"Classification Report:\n{report_classification}")

# ========================== Regression Metrics ==========================

# 1. Mean Absolute Error (MAE)
model_regression = LinearRegression()
scoring_mae = 'neg_mean_absolute_error'
results_mae = cross_val_score(model_regression, X_regression, Y_regression, cv=kfold, scoring=scoring_mae)
print(f"Mean Absolute Error: {results_mae.mean():.3f} ({results_mae.std():.3f})")

# 2. Mean Squared Error (MSE)
scoring_mse = 'neg_mean_squared_error'
results_mse = cross_val_score(model_regression, X_regression, Y_regression, cv=kfold, scoring=scoring_mse)
print(f"Mean Squared Error: {results_mse.mean():.3f} ({results_mse.std():.3f})")

# 3. R2 Metric
scoring_r2 = 'r2'
results_r2 = cross_val_score(model_regression, X_regression, Y_regression, cv=kfold, scoring=scoring_r2)
print(f"R^2: {results_r2.mean():.3f} ({results_r2.std():.3f})")
