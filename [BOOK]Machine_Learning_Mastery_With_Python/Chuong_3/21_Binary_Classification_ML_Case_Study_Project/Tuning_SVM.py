# Tune scaled SVM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
print("%f (%f) with: %r" % (mean, stdev, param))
# Best: 0.867470 using {'kernel': 'rbf', 'C': 1.5}
# 0.758456 (0.099483) with: {'kernel': 'linear', 'C': 0.1}
# 0.529412 (0.118825) with: {'kernel': 'poly', 'C': 0.1}
# 0.573162 (0.130930) with: {'kernel': 'rbf', 'C': 0.1}
# 0.409559 (0.073625) with: {'kernel': 'sigmoid', 'C': 0.1}
# 0.746324 (0.109507) with: {'kernel': 'linear', 'C': 0.3}
# 0.642647 (0.132187) with: {'kernel': 'poly', 'C': 0.3}
# 0.765809 (0.091692) with: {'kernel': 'rbf', 'C': 0.3}
# 0.409559 (0.073625) with: {'kernel': 'sigmoid', 'C': 0.3}
# 0.740074 (0.082636) with: {'kernel': 'linear', 'C': 0.5}
# 0.680147 (0.098595) with: {'kernel': 'poly', 'C': 0.5}
# 0.788235 (0.064190) with: {'kernel': 'rbf', 'C': 0.5}
# 0.409559 (0.073625) with: {'kernel': 'sigmoid', 'C': 0.5}
# 0.746691 (0.084198) with: {'kernel': 'linear', 'C': 0.7}
# 0.740074 (0.127908) with: {'kernel': 'poly', 'C': 0.7}
# 0.812500 (0.085513) with: {'kernel': 'rbf', 'C': 0.7}
# 0.409559 (0.073625) with: {'kernel': 'sigmoid', 'C': 0.7}
# 0.758824 (0.096520) with: {'kernel': 'linear', 'C': 0.9}
# 0.770221 (0.102510) with: {'kernel': 'poly', 'C': 0.9}
# 0.836397 (0.088697) with: {'kernel': 'rbf', 'C': 0.9}
# 0.409559 (0.073625) with: {'kernel': 'sigmoid', 'C': 0.9}
# 0.752574 (0.098883) with: {'kernel': 'linear', 'C': 1.0}
# 0.788235 (0.108418) with: {'kernel': 'poly', 'C': 1.0}
# 0.836397 (0.088697) with: {'kernel': 'rbf', 'C': 1.0}
# 0.409559 (0.073625) with: {'kernel': 'sigmoid', 'C': 1.0}
# 0.769853 (0.106086) with: {'kernel': 'linear', 'C': 1.3}
# 0.818382 (0.107151) with: {'kernel': 'poly', 'C': 1.3}
# 0.848162 (0.080414) with: {'kernel': 'rbf', 'C': 1.3}
# 0.409559 (0.073625) with: {'kernel': 'sigmoid', 'C': 1.3}
# 0.758088 (0.092026) with: {'kernel': 'linear', 'C': 1.5}
# 0.830147 (0.110255) with: {'kernel': 'poly', 'C': 1.5}
# 0.866176 (0.091458) with: {'kernel': 'rbf', 'C': 1.5}
# 0.409559 (0.073625) with: {'kernel': 'sigmoid', 'C': 1.5}
# 0.746324 (0.090414) with: {'kernel': 'linear', 'C': 1.7}
# 0.830515 (0.116706) with: {'kernel': 'poly', 'C': 1.7}
# 0.860294 (0.088281) with: {'kernel': 'rbf', 'C': 1.7}
# 0.409559 (0.073625) with: {'kernel': 'sigmoid', 'C': 1.7}
# 0.758456 (0.094064) with: {'kernel': 'linear', 'C': 2.0}
# 0.830882 (0.108950) with: {'kernel': 'poly', 'C': 2.0}
# 0.866176 (0.095166) with: {'kernel': 'rbf', 'C': 2.0}
# 0.409559 (0.073625) with: {'kernel': 'sigmoid', 'C': 2.0}
