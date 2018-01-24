import pandas as pd
import numpy as np
import os
from utilities import utilities
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV


# Directory to save results
output_directory = './results/introduction/'
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

print("Pre-processing")
# Import data
data = pd.read_csv("data_daily_hourly.csv", delimiter=";", index_col=0)
# Shuffle only once to get same data order for every test - no shuffling later (except CV)
data = shuffle(data, random_state=333)

# Split data into features and targets
X = data.drop(["PM10_next"], axis=1)
y = data["PM10_next"]
# Get names for features output array
column_names = X.columns.values

# Scale to given range - default (0,1)
minmax = MinMaxScaler()
minmax_output = minmax.fit_transform(X)
X_scaled = pd.DataFrame(minmax_output, columns=column_names, index=X.index)

# Split data into train and test without shuffling - test size: 20%
X_train, X_test, y_train, y_test, = train_test_split(X_scaled, y, train_size=0.8, test_size=0.2, shuffle=False)

ridge_reg = RidgeCV(alphas=(50.0, 100.0, 200.0), normalize=False)
dt_reg = DecisionTreeRegressor(max_depth=10)
rf_reg = RandomForestRegressor(n_estimators=15, max_depth=10)
svr_rbf_reg = SVR(kernel='rbf', C=1000, gamma=1e-7)
mlp_reg = MLPRegressor(activation='logistic', hidden_layer_sizes=(100, 300), learning_rate='adaptive')

print('Searching for SVR parameters')
svr_params = [
    {'C': [1, 10, 100], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [1e-8, 1e-7, 1e-6, 1e-5], 'kernel': ['rbf']}
]
svr_grid = GridSearchCV(svr_rbf_reg, svr_params, n_jobs=1, verbose=2)
svr_grid.fit(X_train, y_train)
svr_grid_results = pd.DataFrame(svr_grid.cv_results_)
svr_grid_results = svr_grid_results.sort_values("rank_test_score")
print(svr_grid_results)
svr_grid_results.to_csv(output_directory + 'svr_grid.csv', sep=";")

print('Searching for MLP parameters')
mlp_params = [
    {'hidden_layer_sizes': [(100,), (400,), (100, 100), (400, 100), (400, 100, 50)],
     'activation': ['identity', 'logistic'],
     'alpha': [0.0001, 0.001, 0.01]}
]
mlp_grid = GridSearchCV(mlp_reg, mlp_params, n_jobs=1, verbose=2)
mlp_grid.fit(X_train, y_train)
mlp_grid_results = pd.DataFrame(mlp_grid.cv_results_)
mlp_grid_results = mlp_grid_results.sort_values("rank_test_score")
print(mlp_grid_results)
mlp_grid_results.to_csv(output_directory + 'mlp_grid.csv', sep=";")

print('Searching for RF parameters')
rf_params = [
    {'n_estimators': [10, 15, 17, 20, 25, 30, 50], 'max_depth': [10, 15, 20]}
]
rf_grid = GridSearchCV(rf_reg, rf_params, n_jobs=1, verbose=2)
rf_grid.fit(X_train, y_train)
rf_grid_results = pd.DataFrame(rf_grid.cv_results_)
rf_grid_results = rf_grid_results.sort_values("rank_test_score")
print(rf_grid_results)
rf_grid_results.to_csv(output_directory + 'rf_grid.csv', sep=";")

reg_list = [("RidgeCV", ridge_reg),
            ("DT", dt_reg),
            ("RF", rf_grid.best_estimator_),
            ("SVR_RBF", svr_grid.best_estimator_),
            ("MLP", mlp_grid.best_estimator_)]

print('Testing regressors')
results = utilities.test_regressions(reg_list, X_train, X_test, y_train, y_test, '',
                                     plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
results.to_csv(output_directory + 'results.csv', sep=';')
