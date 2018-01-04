import pandas as pd
from os import listdir
from os.path import isfile, join


my_path = "./raw_data/daily/pollution_1g/"
raw_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

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
del df['Miesiąc.1']
df["PM10_prev"] = df["PM10"].shift(1)
df["PM10_next"] = df["PM10"].shift(-1)
df = df.dropna(axis=0, how="any")

df.index = pd.to_datetime(df.index)

for file in raw_files:
    raw_data_df = pd.read_excel(my_path + file, index_col=0)
    raw_data_df = raw_data_df.iloc[3:]
    raw_data_df.index = pd.to_datetime(raw_data_df.index)
    for index, row in raw_data_df.iterrows():
        print(index.date())
        #print(row)
        #print(row["Pm.a06a"])
        try:
            df.at[index.date(), "PM10_h_" + str(index.hour)] = row["HERE"]
        except:
            print("NOT ADDED:", index.date())

df = df.dropna(axis=0, how="any")
print(df)
df.to_csv("data_proper.csv", sep=";")
