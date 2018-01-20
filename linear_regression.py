import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utilities import utilities
from sklearn.utils import shuffle
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV


print("-----Pre-processing-----")
# Import data
data = pd.read_csv("out.csv", delimiter=";", index_col=0)
data = shuffle(data)

# Split data into features and targets
X = data.drop(["PM10_next"], axis=1)
y = data["PM10_next"]
# Get log of targets
data["PM10_log"] = np.log(data["PM10_next"])
y_log = data["PM10_log"]
# Get names for features output array
column_names = X.columns.values

# Normalize by removing mean and scaling to unit variance
scaler = StandardScaler()
scaler_output = scaler.fit_transform(X)
X_scaled = pd.DataFrame(scaler_output, columns=column_names, index=X.index)

# Split data into train and test with shuffling - test size: 20%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, shuffle=False)
# Split data with log(target) into train and test with shuffling - test size: 20%
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_scaled, y_log, train_size=0.8, shuffle=False)

print("-----Feature selection - RFE-----")
# Get simple linear regression object
ridge_reg = Ridge()
# Feature selection - done with recursive feature elimination
feature_selection_rfe = RFECV(ridge_reg, step=1, verbose=0, cv=ShuffleSplit(n_splits=10, train_size=0.8))
feature_selection_rfe.fit(X_train, y_train)
# Get new data set containing only selected features
X_selected = X_scaled[X_scaled.columns[feature_selection_rfe.get_support()]]
# Print feature selection results
print("RFE, number of selected features:", feature_selection_rfe.n_features_)
# put here some plots
X_train_fs_rfe, X_test_fs_rfe, y_train_fs_rfe, y_test_fs_rfe = train_test_split(X_selected, y, train_size=0.8, shuffle=False)

# Model selection by grid search - try every provided parameters set
parameters_gs = {
    'alpha': [100, 200, 500, 1000, 2000],
    'solver': ['auto', 'lsqr', 'saga'],
}
grid_rf = GridSearchCV(ridge_reg, parameters_gs, cv=2, verbose=0, scoring='r2', return_train_score=True)
grid_rf.fit(X_train, y_train)

# Print results of model selection
grid_result_df = pd.DataFrame(grid_rf.cv_results_)
grid_result_df.to_csv("result.csv", sep=";")
print(grid_result_df.sort_values('rank_test_score'))

# Perform Test with basic model
print("-----Ridge test-----")
utilities.test_regression(ridge_reg, X_train, X_test, y_train, y_test)
print("-----Ridge test with log(target)-----")
utilities.test_regression(ridge_reg, X_train_log, X_test_log, y_train_log, y_test_log)
print("-----Ridge test on selected features-----")
utilities.test_regression(ridge_reg, X_train_fs_rfe, X_test_fs_rfe, y_train_fs_rfe, y_test_fs_rfe)

# Perform test on GridSearch selected model
print("-----Ridge test-----")
utilities.test_regression(grid_rf, X_train, X_test, y_train, y_test)
print("-----Ridge test with log(target)-----")
utilities.test_regression(grid_rf, X_train_log, X_test_log, y_train_log, y_test_log)
print("-----Ridge test on selected features-----")
utilities.test_regression(grid_rf, X_train_fs_rfe, X_test_fs_rfe, y_train_fs_rfe, y_test_fs_rfe)

