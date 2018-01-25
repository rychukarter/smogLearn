import pandas as pd
import numpy as np
import os
from utilities import utilities
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import f_regression, RFECV, SelectKBest, SelectFromModel
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from sklearn.decomposition import PCA

# Directory to save results
output_directory = './results/all_in_one/second/scaled_minmax/'
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
# Get log of targets
data["PM10_log"] = np.log(data["PM10_next"])
y_log = data["PM10_log"]
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
X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(X_selected, y, y_log, train_size=0.8,
                                                                             test_size=0.2, shuffle=True)

print("Performing feature number reduction")
print("Performing PCA")
# Perform PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled_std)
X_train_pca, X_test_pca = train_test_split(X_pca, train_size=0.8, test_size=0.2, shuffle=False)

# Get simple linear regression objects for RFE and SFM
ridge_reg = Ridge(normalize=False)
# Get Dataframe for selected features
selected_features_df = pd.DataFrame(X.columns, columns=['features'])
print("Feature selection - RFE")
# Feature selection - done with recursive feature elimination
feature_selection_rfe = RFECV(ridge_reg, step=10, verbose=0, cv=ShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2))
feature_selection_rfe.fit(X_train, y_train)
# Get new data set containing only selected features
X_selected_rfe = X_selected[X_selected.columns[feature_selection_rfe.get_support()]]
# Save feature selection results
selected_features_df['RFE'] = selected_features_df.apply(lambda x: 'X' if x['features'] in X_selected_rfe.columns else '',
                                                         axis=1)
# put here some plots
X_train_fs_rfe, X_test_fs_rfe = train_test_split(X_selected_rfe, train_size=0.8, test_size=0.2, shuffle=False)

print("Feature selection - SelectKBest")
# Feature selection by selecting K best features
feature_selection_skb = SelectKBest(score_func=f_regression, k=150)
feature_selection_skb.fit(X_train, y_train)
# Get new data set containing only selected features
X_selected_skb = X_selected[X_selected.columns[feature_selection_skb.get_support()]]
# Save feature selection results
selected_features_df['SelectKBest'] = selected_features_df.apply(lambda x: 'X' if x['features'] in X_selected_skb.columns else '',
                                                                 axis=1)
X_train_fs_skb, X_test_fs_skb = train_test_split(X_selected_skb, train_size=0.8, test_size=0.2, shuffle=False)

print("Feature selection - SelectFromModel")
# Feature selection by threshold
feature_selection_sfm = SelectFromModel(ridge_reg, threshold="median")
feature_selection_sfm.fit(X_train, y_train)
# Get new data set containing only selected features
X_selected_sfm = X_selected[X_selected.columns[feature_selection_sfm.get_support()]]
# Save feature selection results
selected_features_df['SelectFromModel'] = selected_features_df.apply(lambda x: 'X' if x['features'] in X_selected_sfm.columns else '',
                                                                     axis=1)
X_train_fs_sfm, X_test_fs_sfm = train_test_split(X_selected_sfm, train_size=0.8, test_size=0.2, shuffle=False)

# Save feature selection results
selected_features_df = selected_features_df.set_index('features')
selected_features_df.to_csv(output_directory + 'selected_features.csv', sep=';')


# Get regressors objects
print('Getting regressors')
ridge_reg_cv = RidgeCV(alphas=(50.0, 100.0, 200.0), normalize=False)
lasso_reg_cv = LassoCV(normalize=False, n_alphas=10)
svr_rbf_reg = SVR(kernel='rbf')
svr_linear_reg = SVR(kernel='linear')
mlp_reg = MLPRegressor(learning_rate='adaptive')
dt_reg = DecisionTreeRegressor()
et_reg = ExtraTreesRegressor()
rf_reg = RandomForestRegressor()
knn_reg = KNeighborsRegressor()

# Tuning hyper-parameters of estimators with cross-validated GridSearch
print('Searching for SVR with linear kernel parameters')
svr_params = [
    {'C': [1, 10, 15, 50], 'kernel': ['linear']}
]
svr_linear_grid = GridSearchCV(svr_linear_reg, svr_params, n_jobs=1, verbose=0, return_train_score=True)
svr_linear_grid.fit(X_train, y_train)
svr_linear_grid_results = pd.DataFrame(svr_linear_grid.cv_results_)
svr_linear_grid_results = svr_linear_grid_results.sort_values("rank_test_score")
svr_linear_grid_results.to_csv(output_directory + 'svr_linear_grid.csv', sep=";")

print('Searching for SVR with rbf kernel parameters')
svr_params = [
    {'C': [100, 500, 1000], 'gamma': [1e-6, 0.5e-5, 1e-5, 1e-4], 'kernel': ['rbf']}
]
svr_rbf_grid = GridSearchCV(svr_rbf_reg, svr_params, n_jobs=1, verbose=0, return_train_score=True)
svr_rbf_grid.fit(X_train, y_train)
svr_rbf_grid_results = pd.DataFrame(svr_rbf_grid.cv_results_)
svr_rbf_grid_results = svr_rbf_grid_results.sort_values("rank_test_score")
svr_rbf_grid_results.to_csv(output_directory + 'svr_rbf_grid.csv', sep=";")

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

print('Searching for DT parameters')
dt_params = [
    {'max_depth': [3, 4, 5]}
]
dt_grid = GridSearchCV(dt_reg, dt_params, n_jobs=1, verbose=0, return_train_score=True)
dt_grid.fit(X_train, y_train)
dt_grid_results = pd.DataFrame(dt_grid.cv_results_)
dt_grid_results = dt_grid_results.sort_values("rank_test_score")
dt_grid_results.to_csv(output_directory + 'dt_grid.csv', sep=";")

print('Searching for ET parameters')
et_params = [
    {'n_estimators': [10, 20, 50, 100], 'max_depth': [10, 15, 20]}
]
et_grid = GridSearchCV(et_reg, et_params, n_jobs=1, verbose=0, return_train_score=True)
et_grid.fit(X_train, y_train)
et_grid_results = pd.DataFrame(et_grid.cv_results_)
et_grid_results = et_grid_results.sort_values("rank_test_score")
et_grid_results.to_csv(output_directory + 'et_grid.csv', sep=";")

print('Searching for RF parameters')
rf_params = [
    {'n_estimators': [25, 30, 50, 100], 'max_depth': [10, 15, 20]}
]
rf_grid = GridSearchCV(rf_reg, rf_params, n_jobs=1, verbose=0, return_train_score=True)
rf_grid.fit(X_train, y_train)
rf_grid_results = pd.DataFrame(rf_grid.cv_results_)
rf_grid_results = rf_grid_results.sort_values("rank_test_score")
rf_grid_results.to_csv(output_directory + 'rf_grid.csv', sep=";")

print('Searching for KNN parameters')
knn_params = [
    {'n_neighbors': [10, 15, 20, 50]}
]
knn_grid = GridSearchCV(knn_reg, knn_params, n_jobs=1, verbose=0, return_train_score=True)
knn_grid.fit(X_train, y_train)
knn_grid_results = pd.DataFrame(knn_grid.cv_results_)
knn_grid_results = knn_grid_results.sort_values("rank_test_score")
knn_grid_results.to_csv(output_directory + 'knn_grid.csv', sep=";")


# Get regressors list for testing
reg_list = [("RidgeCV", ridge_reg_cv),
            ("LassoCV", lasso_reg_cv),
            ("SVR_linear", svr_linear_grid.best_estimator_),
            ("SVR_rbf", svr_rbf_grid.best_estimator_),
            ("MLP", mlp_grid.best_estimator_),
            ("DT", dt_grid.best_estimator_),
            ("ET", et_grid.best_estimator_),
            ("RF", rf_grid.best_estimator_),
            ("KNN", knn_grid.best_estimator_)]

# Define testing
plot_lc = True
plot_hist = True
n = 5
# Test all regressor with all data combinations
print("Perform test - basic data")
results = utilities.test_regressions_n(reg_list, X_selected, y, n, '', plot_learning_curves=plot_lc,
                                       plot_histogram=plot_hist, save_path=output_directory)
results_log = utilities.test_regressions_n(reg_list, X_selected, y_log, n, '_log', plot_learning_curves=plot_lc,
                                           plot_histogram=plot_hist, save_path=output_directory)
print("Perform test - PCA")
results_pca = utilities.test_regressions_n(reg_list, X_pca, y, n, '_pca', plot_learning_curves=plot_lc,
                                           plot_histogram=plot_hist, save_path=output_directory)
results_pca_log = utilities.test_regressions_n(reg_list, X_pca, y_log, n, '_pca_log', plot_learning_curves=plot_lc,
                                               plot_histogram=plot_hist, save_path=output_directory)
print("Perform test - RFE")
results_fs_rfe = utilities.test_regressions_n(reg_list, X_selected_rfe, y, n, '_rfe', plot_learning_curves=plot_lc,
                                              plot_histogram=plot_hist, save_path=output_directory)
results_fs_rfe_log = utilities.test_regressions_n(reg_list, X_selected_rfe, y_log, n, '_rfe_log', plot_learning_curves=plot_lc,
                                                  plot_histogram=plot_hist, save_path=output_directory)
print("Perform test - SKB")
results_fs_skb = utilities.test_regressions_n(reg_list, X_selected_skb, y, n, '_skb', plot_learning_curves=plot_lc,
                                              plot_histogram=plot_hist, save_path=output_directory)
results_fs_skb_log = utilities.test_regressions_n(reg_list, X_selected_skb, y_log, n, '_skb_log', plot_learning_curves=plot_lc,
                                                  plot_histogram=plot_hist, save_path=output_directory)
print("Perform test - SFM")
results_fs_sfm = utilities.test_regressions_n(reg_list, X_selected_sfm, y, n, '_sfm', plot_learning_curves=plot_lc,
                                              plot_histogram=plot_hist, save_path=output_directory)
results_fs_sfm_log = utilities.test_regressions_n(reg_list, X_selected_sfm, y_log, n, '_sfm_log', plot_learning_curves=plot_lc,
                                                  plot_histogram=plot_hist, save_path=output_directory)
# Save results
out_df = pd.concat([results, results_pca, results_fs_rfe, results_fs_skb, results_fs_sfm,
                    results_log, results_pca_log, results_fs_rfe_log, results_fs_skb_log, results_fs_sfm_log])
out_df.to_csv(output_directory + 'results.csv', sep=';')
