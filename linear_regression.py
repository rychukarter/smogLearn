import pandas as pd
import numpy as np
import os
from utilities import utilities
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.feature_selection import f_regression, RFECV, SelectKBest, SelectFromModel
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.decomposition import PCA

# Directory to save results
output_directory = './results/linear/scaled_std/'
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
X_selected = X_scaled_std
# Split data into train and test without shuffling - test size: 20%
X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(X_selected, y, y_log,
                                                                             train_size=0.8, test_size=0.2,
                                                                             shuffle=False)

print("Performing PCA")
# Perform PCA
pca = PCA(n_components=200)
X_pca = pca.fit_transform(X_scaled_std)
X_train_pca, X_test_pca = train_test_split(X_pca, train_size=0.8, test_size=0.2, shuffle=False)


print("Getting estimators")
# Get simple linear regression objects
ridge_reg = Ridge(normalize=False)
ridge_reg_cv = RidgeCV(alphas=(50.0, 100.0, 200.0), normalize=False)
lasso_reg = Lasso(normalize=False)
lasso_reg_cv = LassoCV(normalize=False, n_alphas=10)
reg_list = [("Ridge", ridge_reg),
            ("RidgeCV", ridge_reg_cv),
            ("Lasso", lasso_reg),
            ("LassoCV", lasso_reg_cv)]


print("Feature selection - RFE")
# Feature selection - done with recursive feature elimination
feature_selection_rfe = RFECV(ridge_reg, step=1, verbose=0, cv=ShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2,
                                                                            random_state=2973))
feature_selection_rfe.fit(X_train, y_train)
# Get new data set containing only selected features
X_selected_rfe = X_selected[X_selected.columns[feature_selection_rfe.get_support()]]
# Print feature selection results
print("RFE, selected features:", len(X_selected_rfe.columns))
# put here some plots
X_train_fs_rfe, X_test_fs_rfe = train_test_split(X_selected_rfe, train_size=0.8, test_size=0.2, shuffle=False)


print("Feature selection - SelectKBest")
# Feature selection by selecting K best features
feature_selection_skb = SelectKBest(score_func=f_regression, k=150)
feature_selection_skb.fit(X_train, y_train)
# Get new data set containing only selected features
X_selected_skb = X_selected[X_selected.columns[feature_selection_skb.get_support()]]
# Print feature selection results
print("SelectKBest, selected features:", len(X_selected_skb.columns))
X_train_fs_skb, X_test_fs_skb = train_test_split(X_selected_skb, train_size=0.8, test_size=0.2, shuffle=False)


print("Feature selection - SelectFromModel")
# Feature selection by threshold
feature_selection_sfm = SelectFromModel(ridge_reg_cv, threshold="median")
feature_selection_sfm.fit(X_train, y_train)
# Get new data set containing only selected features
X_selected_sfm = X_selected[X_selected.columns[feature_selection_sfm.get_support()]]
# Print feature selection results
print("SelectFromModel, selected features:", len(X_selected_sfm.columns))
X_train_fs_sfm, X_test_fs_sfm = train_test_split(X_selected_sfm, train_size=0.8, test_size=0.2, shuffle=False)


print("Perform test - basic data")
results = utilities.test_regressions(reg_list, X_train, X_test, y_train, y_test, '',
                                     plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
results_log = utilities.test_regressions(reg_list, X_train, X_test, y_train_log, y_test_log, '_log',
                                         plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
print("Perform test - PCA")
results_pca = utilities.test_regressions(reg_list, X_train_pca, X_test_pca, y_train, y_test, '_pca',
                                         plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
results_pca_log = utilities.test_regressions(reg_list, X_train_pca, X_test_pca, y_train_log, y_test_log, '_pca_log',
                                             plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
print("Perform test - RFE")
results_fs_rfe = utilities.test_regressions(reg_list, X_train_fs_rfe, X_test_fs_rfe, y_train, y_test, '_rfe',
                                            plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
results_fs_rfe_log = utilities.test_regressions(reg_list, X_train_fs_rfe, X_test_fs_rfe, y_train_log, y_test_log,
                                                '_rfe_log', plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
print("Perform test - SKB")
results_fs_skb = utilities.test_regressions(reg_list, X_train_fs_skb, X_test_fs_skb, y_train, y_test, '_skb',
                                            plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
results_fs_skb_log = utilities.test_regressions(reg_list, X_train_fs_skb, X_test_fs_skb, y_train_log, y_test_log,
                                                '_skb_log', plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
print("Perform test - SFM")
results_fs_sfm = utilities.test_regressions(reg_list, X_train_fs_sfm, X_test_fs_sfm, y_train, y_test, '_sfm',
                                            plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
results_fs_sfm_log = utilities.test_regressions(reg_list, X_train_fs_sfm, X_test_fs_sfm, y_train_log, y_test_log,
                                                '_sfm_log', plot_learning_curves=True, plot_histogram=True, save_path=output_directory)

out_df = pd.concat([results, results_pca, results_fs_rfe, results_fs_skb, results_fs_sfm,
                    results_log, results_pca_log, results_fs_rfe_log, results_fs_skb_log, results_fs_sfm_log])
out_df.to_csv(output_directory + 'results.csv', sep=';')
