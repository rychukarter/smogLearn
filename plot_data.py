import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data_proper.csv", delimiter=";", index_col=0)
df = df.reset_index(drop=True)
print(df.columns)

x_axis = df.index

plt.plot(x_axis[2700:3200], df["PM10_h_12"][2700:3200],  label="y_test")
plt.plot(x_axis[2700:3200], df["PM10_h_2"][2700:3200], label="y_test")
plt.plot(x_axis[2700:3200], df["PM10_h_6"][2700:3200], label="y_test")
plt.plot(x_axis[2700:3200], df["PM10_h_18"][2700:3200], label="y_test")
plt.legend()
plt.grid('True')
plt.show()
