# KNN Algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
print("%f (%f) with: %r" % (mean, stdev, param))
# Best: -18.172137 using {'n_neighbors': 3}
# -20.169640 (14.986904) with: {'n_neighbors': 1}
# -18.109304 (12.880861) with: {'n_neighbors': 3}
# -20.063115 (12.138331) with: {'n_neighbors': 5}
# -20.514297 (12.278136) with: {'n_neighbors': 7}
# -20.319536 (11.554509) with: {'n_neighbors': 9}
# -20.963145 (11.540907) with: {'n_neighbors': 11}
# -21.099040 (11.870962) with: {'n_neighbors': 13}
# -21.506843 (11.468311) with: {'n_neighbors': 15}
# -22.739137 (11.499596) with: {'n_neighbors': 17}
# -23.829011 (11.277558) with: {'n_neighbors': 19}
# -24.320892 (11.849667) with: {'n_neighbors': 21}
