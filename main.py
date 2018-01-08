import pandas as pd
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('out.csv', delimiter=";", index_col=0)

df = df.reset_index(drop=True)
df = shuffle(df)

msk = np.random.rand(len(df)) < 0.8

train = df[msk]
test = df[~msk]
#train = shuffle(train)

X_train = train.drop(["PM10_next"], axis=1)
y_train = train["PM10_next"]

X_test = test.drop(["PM10_next"], axis=1)
y_test = test["PM10_next"]

parameters = {
    "kernel": ["rbf"],
    "C": [10, 100, 1000],
    "gamma": [1e-8, 1e-7, 1e-6, 1e-5]
    }

grid = GridSearchCV(SVR(), parameters, cv=5, verbose=2, return_train_score=True)
regr_1 = DecisionTreeRegressor(max_depth=10)
regr_rf = RandomForestRegressor(max_depth=10, random_state=3)

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)


#svr_rbf = SVR(kernel='rbf', C=1e6, gamma=1e-8)
#svr_lin = SVR(kernel='linear', C=1e4)
#svr_poly = SVR(kernel='poly', C=1e4, degree=2)

grid_rbf = grid.fit(X_train, y_train)
tree_reg = regr_1.fit(X_train, y_train)
forest_reg = regr_rf.fit(X_train, y_train)

y_rbf = grid_rbf.predict(X_test)
tree_v = tree_reg.predict(X_test)
forest_v = forest_reg.predict(X_test)
print(grid.cv_results_)
print(y_rbf[0:10], y_test.iloc[0:10])
print(r2_score(y_test, y_rbf))
print(mean_squared_error(y_test, y_rbf))
print(mean_absolute_error(y_test, y_rbf))
print("----------------------------------------")
print(tree_v[0:10], y_test.iloc[0:10])
print(r2_score(y_test, tree_v))
print(mean_squared_error(y_test, tree_v))
print(mean_absolute_error(y_test, tree_v))
print("----------------------------------------")
print(forest_v[0:10], y_test.iloc[0:10])
print(r2_score(y_test, forest_v))
print(mean_squared_error(y_test, forest_v))
print(mean_absolute_error(y_test, forest_v))

#y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
#print(r2_score(y_test, y_lin))
#y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
#print(r2_score(y_test, y_poly))

# df.to_csv("data2.csv", sep=";")

x_axis = np.arange(100)
plt.plot(x_axis, y_test.iloc[0:100], label="y_test")
plt.plot(x_axis, forest_v[0:100], label="forest_v")
plt.legend()
plt.grid('True')
plt.show()
plt.plot(x_axis, y_rbf[0:100], label="y_rbf")
plt.show()
