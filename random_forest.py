import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utilities import utilities
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Import data
data = pd.read_csv("out.csv", delimiter=";", index_col=0)

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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, shuffle=True)

# Get default initiated random forest regressor object
estimator_rf = RandomForestRegressor()

# Make prediction on all features with default regressor
estimator_rf.fit(X_train, y_train)
y_rf = estimator_rf.predict(X_test)
y_rf_train = estimator_rf.predict(X_train)
print("MAE:", mean_absolute_error(y_test, y_rf))
print("R2:", r2_score(y_test, y_rf))

# Plot learning curves
learning_curve = utilities.plot_learning_curve(estimator_rf, "Random Forest", X_train, y_train)
learning_curve.show()

# Feature selection - done with recursive feature elimination
feature_selection_rfe = RFECV(estimator_rf, step=10, verbose=2,
                              cv=ShuffleSplit(n_splits=5, train_size=0.8), scoring="r2")
feature_selection_rfe.fit(X_train, y_train)

# Get new data set containing only selected features
feature_mask = feature_selection_rfe.get_support()
X_selected = X_scaled[X_scaled.columns[feature_mask]]

# Print feature selection results
print("Number of selected features:", feature_selection_rfe.n_features_)
print("Feature selection grid scores:", feature_selection_rfe.grid_scores_)
print("Selected features list:", X_selected.columns)

# Model selection by grid search - try every provided parameters set
parameters_gs = {
    'n_estimators': [5, 7, 10, 12, 15], #how many trees in the forest
    'max_depth': [2, 5, 10, 15, 20]     #how deep can trees get
}
grid_rf = GridSearchCV(estimator_rf, parameters_gs, cv=2, verbose=2, scoring='r2', return_train_score=True)
grid_rf.fit(X_train, y_train)

# Print results of model selection
grid_result_df = pd.DataFrame(grid_rf.cv_results_)
grid_result_df.to_csv("result.csv", sep=";")
print(grid_result_df.sort_values('rank_test_score'))
print("Best score:", grid_rf.best_score_)
