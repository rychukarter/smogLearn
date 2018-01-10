import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
# Import data
data = pd.read_csv("out.csv", delimiter=";", index_col=0)

# Split data into features and targets
X = data.drop(["PM10_next"], axis=1)
y = data["PM10_next"]

# Get names for features output array
names = X.columns.values

scaler = StandardScaler()
X = scaler.fit_transform(X)


estimator = RandomForestRegressor(n_estimators=20, max_depth=15)
#estimator = MLPRegressor(hidden_layer_sizes=(100, 100))
rfe = RFECV(estimator, step=1, verbose=2, cv=ShuffleSplit(n_splits=5, train_size=0.6), scoring="r2")
rfe.fit(X, y)
print(rfe.grid_scores_)
df = pd.DataFrame()
df["names"] = names
df["ranking"] = rfe.ranking_
df["grid_score"] = rfe.grid_scores_
df["support"] = rfe.support_

print("Number of selected features:", rfe.n_features_)
print("***------------------------------------------------------***")
print(df.sort_values("grid_score"))