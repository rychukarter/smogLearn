import pandas as pd
import matplotlib.pyplot as plt

# Import Data
df = pd.read_csv("data_daily_hourly.csv", delimiter=";", index_col=0)
df.index = pd.to_datetime(df.index)

# Get PM10 values to plot
df2 = df["PM10"]
# Plot and show
df2.plot()
plt.show()
