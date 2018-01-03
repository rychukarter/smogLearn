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

    print(raw_data_df.columns)



#df.to_csv("data4.csv", sep=";")
