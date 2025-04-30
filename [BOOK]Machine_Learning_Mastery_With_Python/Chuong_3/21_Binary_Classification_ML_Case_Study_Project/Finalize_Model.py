# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C=1.5)
model.fit(rescaledX, Y_train)
# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
# 0.857142857143
# [[23 4]
# [ 2 13]]
# precision recall f1-score support
# M 0.92 0.85 0.88 27
# R 0.76 0.87 0.81 15
# avg / total 0.86 0.86 0.86 42
