import pandas as pd
import numpy as np
import os
from utilities import utilities
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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
# Get log of targets
data["PM10_log"] = np.log(data["PM10_next"])
y_log = data["PM10_log"]
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
rf_reg = RandomForestRegressor(n_estimators=15, max_depth=20)
svr_rbf_reg = SVR(kernel='rbf', C=1000, gamma=1e-7)
mlp_reg = MLPRegressor(activation='logistic')

reg_list = [("RidgeCV", ridge_reg),
            ("DT", dt_reg),
            ("RF", rf_reg),
            ("SVR_RBF", svr_rbf_reg),
            ("MLP", mlp_reg)]

results = utilities.test_regressions(reg_list, X_train, X_test, y_train, y_test, '',
                                     plot_learning_curves=True, plot_histogram=True, save_path=output_directory)
results.to_csv(output_directory + 'results.csv', sep=';')
