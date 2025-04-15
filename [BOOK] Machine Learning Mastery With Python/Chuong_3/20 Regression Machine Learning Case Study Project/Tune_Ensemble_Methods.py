# Tune scaled GBM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
print("%f (%f) with: %r" % (mean, stdev, param))
# Best: -9.356471 using {'n_estimators': 400}
# -10.794196 (4.711473) with: {'n_estimators': 50}
# -10.023378 (4.430026) with: {'n_estimators': 100}
# -9.677657 (4.264829) with: {'n_estimators': 150}
# -9.523680 (4.259064) with: {'n_estimators': 200}
# -9.432755 (4.250884) with: {'n_estimators': 250}
# -9.414258 (4.262219) with: {'n_estimators': 300}
# -9.353381 (4.242264) with: {'n_estimators': 350}
# -9.339880 (4.255717) with: {'n_estimators': 400}
