import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data_daily_hourly.csv", delimiter=";", index_col=0)

df.index = pd.to_datetime(df.index)
df2 = df["PM10"]
df2.plot()
plt.show()
