import pandas as pd
from os import listdir
from os.path import isfile, join


my_path = "./raw_data/"
raw_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

df = pd.read_csv("data3.csv", delimiter=";", index_col=0)
del df['Miesiąc.1']
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
            df.at[index.date(),"PM10_h_" + str(index.hour)] = row["MzWarNiepodKom"]
        except:
            print("NOT ADDED:", index.date())

print(df)
df.to_csv("test.csv", sep=";")


#df.to_csv("data4.csv", sep=";")
