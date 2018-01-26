import pandas as pd
import os
from utilities import utilities
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA

# Directory to save results
output_directory = './results/ensemble/second/scaled_minmax/'
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

print("Pre-processing")
# Import data
data = pd.read_csv("data_daily_hourly.csv", delimiter=";", index_col=0)
# Shuffle only once to get same data order for every test - no shuffling later (except CV)
data = shuffle(data)

# Split data into features and targets
X = data.drop(["PM10_next"], axis=1)
y = data["PM10_next"]
# Get names for features output array
column_names = X.columns.values

# Scale by removing mean and scaling to unit variance
scaler = StandardScaler()
scaler_output = scaler.fit_transform(X)
X_scaled_std = pd.DataFrame(scaler_output, columns=column_names, index=X.index)

# Scale to given range - default (0,1)
minmax = MinMaxScaler()
minmax_output = minmax.fit_transform(X)
X_scaled_minmax = pd.DataFrame(minmax_output, columns=column_names, index=X.index)

# Select between Normalization and Scaling
X_selected = X_scaled_minmax
# Split data into train and test without shuffling - test size: 20%
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, train_size=0.8, test_size=0.2, shuffle=True)

print("Performing feature number reduction")
print("Performing PCA")
# Perform PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled_std)
X_train_pca, X_test_pca = train_test_split(X_pca, train_size=0.8, test_size=0.2, shuffle=False)

# Get regressors objects
print('Getting regressors')
ridge_reg_cv = RidgeCV(alphas=(50.0, 100.0, 200.0), normalize=False)
svr_linear_reg = SVR(kernel='linear')
mlp_reg = MLPRegressor(learning_rate='adaptive')
rf_reg = RandomForestRegressor()

# Tuning hyper-parameters of estimators with cross-validated GridSearch
print('Searching for SVR with linear kernel parameters')
svr_params = [
    {'C': [1, 10, 15], 'kernel': ['linear']}
]
svr_linear_grid = GridSearchCV(svr_linear_reg, svr_params, n_jobs=1, verbose=0, return_train_score=True)
svr_linear_grid.fit(X_train, y_train)
svr_linear_grid_results = pd.DataFrame(svr_linear_grid.cv_results_)
svr_linear_grid_results = svr_linear_grid_results.sort_values("rank_test_score")
svr_linear_grid_results.to_csv(output_directory + 'svr_linear_grid.csv', sep=";")

print('Searching for MLP parameters')
mlp_params = [
    {'hidden_layer_sizes': [(50, 100), (100, 100), (400, 100), (200, 100, 50)],
     'activation': ['identity'],
     'alpha': [0.001, 0.005, 0.01]}
]
mlp_grid = GridSearchCV(mlp_reg, mlp_params, n_jobs=1, verbose=0, return_train_score=True)
mlp_grid.fit(X_train, y_train)
mlp_grid_results = pd.DataFrame(mlp_grid.cv_results_)
mlp_grid_results = mlp_grid_results.sort_values("rank_test_score")
mlp_grid_results.to_csv(output_directory + 'mlp_grid.csv', sep=";")

# Do ensemble
ensemble_list = [("RidgeCV", ridge_reg_cv),
                 ("SVR_linear", svr_linear_grid.best_estimator_),
                 ("MLP", mlp_grid.best_estimator_)]

regressors_output = pd.DataFrame()
for name, e in ensemble_list:
    print(e)
    e.fit(X_selected, y)
    out = e.predict(X_selected)
    regressors_output[name] = out

regressors_output_pca = pd.DataFrame()
for name, e in ensemble_list:
    e.fit(X_pca, y)
    out = e.predict(X_pca)
    regressors_output_pca[name] = out

print('Searching for RF parameters')
rf_params = [
    {'n_estimators': [25, 30, 50, 100], 'max_depth': [10, 15, 20]}
]
rf_grid = GridSearchCV(rf_reg, rf_params, n_jobs=1, verbose=0, return_train_score=True)
rf_grid.fit(regressors_output, y)
rf_grid_results = pd.DataFrame(rf_grid.cv_results_)
rf_grid_results = rf_grid_results.sort_values("rank_test_score")
rf_grid_results.to_csv(output_directory + 'rf_grid.csv', sep=";")

# Get regressors list for testing
reg_list = [("RF", rf_grid.best_estimator_)]

# Define testing
plot_lc = True
plot_hist = True
n = 5
# Test all regressor with all data combinations
print("Perform test - basic data")
results = utilities.test_regressions_n(reg_list, regressors_output, y, n, '', plot_learning_curves=plot_lc,
                                       plot_histogram=plot_hist, save_path=output_directory)
print("Perform test - PCA")
results_pca = utilities.test_regressions_n(reg_list, regressors_output_pca, y, n, '_pca', plot_learning_curves=plot_lc,
                                           plot_histogram=plot_hist, save_path=output_directory)

# Save results
out_df = pd.concat([results, results_pca])
out_df.to_csv(output_directory + 'results.csv', sep=';')
