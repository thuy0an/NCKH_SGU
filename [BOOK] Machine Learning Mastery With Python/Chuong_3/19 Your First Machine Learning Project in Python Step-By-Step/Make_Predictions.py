# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
# 0.9
# [[ 7 0 0]
# [ 0 11 1]
# [ 0 2 9]]

#              precision   recall   f1-score   support
# Iris-setosa     1.00       1.00     1.00       7
# Iris-versicolor 0.85       0.92     0.88       12
# Iris-virginica  0.90       0.82     0.86       11
# avg / total     0.90       0.90     0.90       30
