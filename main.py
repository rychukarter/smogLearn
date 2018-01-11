import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('out.csv', delimiter=";", index_col=0)
df = df.reset_index(drop=True)
df = shuffle(df)

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

X_train = train.drop(["PM10_next"], axis=1)
y_train = train["PM10_next"]
X_test = test.drop(["PM10_next"], axis=1)
y_test = test["PM10_next"]


regr_1 = DecisionTreeRegressor(max_depth=10)
regr_rf = RandomForestRegressor(n_estimators=15, max_depth=20)
regr_lin = LinearRegression()
svr_rbf = SVR(kernel='rbf', C=1000, gamma=1e-7)

svr_rbf = svr_rbf.fit(X_train, y_train)
tree_reg = regr_1.fit(X_train, y_train)
forest_reg = regr_rf.fit(X_train, y_train)
rand_lin = regr_lin.fit(X_train, y_train)

rbf_v = svr_rbf.predict(X_test)
tree_v = tree_reg.predict(X_test)
forest_v = forest_reg.predict(X_test)
lin_v = rand_lin.predict(X_test)


print(r2_score(y_test, rbf_v))
print(mean_squared_error(y_test, rbf_v))
print(mean_absolute_error(y_test, rbf_v))
print("----------------------------------------")
print(r2_score(y_test, tree_v))
print(mean_squared_error(y_test, tree_v))
print(mean_absolute_error(y_test, tree_v))
print("----------------------------------------")
print(r2_score(y_test, forest_v))
print(mean_squared_error(y_test, forest_v))
print(mean_absolute_error(y_test, forest_v))
print("----------------------------------------")
print(r2_score(y_test, lin_v))
print(mean_squared_error(y_test, lin_v))
print(mean_absolute_error(y_test, lin_v))

x_axis = np.arange(50)
plt.plot(x_axis, y_test.iloc[0:50], label="y_test")
plt.plot(x_axis, forest_v[0:50], label="forest_v")
plt.legend()
plt.grid('True')
plt.show()
