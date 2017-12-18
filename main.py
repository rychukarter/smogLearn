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
import matplotlib as mpl


'''
df = pd.read_csv("data2.csv", delimiter=";", index_col=0)
df.index = pd.to_datetime(df.index)
df2 = pd.read_excel("./raw_data/daily/pollution/2015_PM10_24g.xlsx", index_col=0)
df2 = df2.iloc[2:]
df2.index = pd.to_datetime(df2.index)
df.loc["2015-01-01":"2015-12-31", 'PM10'] = df2['PM10']
df['PM10'] = df['PM10'].str.replace(',', '.')
df['PM10'] = df['PM10'].astype(float)
empty_rows = df.isnull().sum()
del df["Stan gruntu Z/R"]
del df["Rodzaj opadu"]
df["PM10_prev"] = df["PM10"].shift(1)
df = df.dropna(axis=0, how="any")
'''
#df.to_csv("data3.csv", sep=";")

df = pd.read_csv("data3.csv", delimiter=";", index_col=0)
#df.index = pd.to_datetime(df.index)
del df['Miesiąc.1']
df = df.reset_index(drop=True)
#df = shuffle(df)

msk = np.random.rand(len(df)) < 0.8

train = df[msk]
test = df[~msk]
train = shuffle(train)

X_train = train.drop(["PM10"], axis=1)
y_train = train["PM10"]

X_test = test.drop(["PM10"], axis=1)
y_test = test["PM10"]

parameters = {
    "kernel": ["rbf"],
    "C": [10, 100, 1000],
    "gamma": [1e-8, 1e-7, 1e-6, 1e-5]
    }

grid = GridSearchCV(SVR(), parameters, cv=5, verbose=2)
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


