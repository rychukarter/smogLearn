import pandas as pd
from os import listdir
from os.path import isfile, join

my_path = "./raw_data/daily/pollution_1g/"
raw_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

df = pd.read_csv("data2.csv", delimiter=";", index_col=0)
df.index = pd.to_datetime(df.index)
print(df.columns)
df2 = pd.read_excel("./raw_data/daily/pollution/2015_PM10_24g.xlsx", index_col=0)
df2 = df2.iloc[2:]
df2.index = pd.to_datetime(df2.index)
df['PM10'] = df['PM10'].str.replace(',', '.')
df.loc["2015-01-01":"2015-12-31", 'PM10'] = df2['PM10']
df['PM10'] = df['PM10'].astype(float)
del df["Stan gruntu Z/R"]
del df["Rodzaj opadu"]
del df['Miesiąc.1']
del df['Wystąpienie błyskawicy']
del df['Średnia dobowa temperatura']
df["PM10_prev"] = df["PM10"].shift(1)
df["PM10_next"] = df["PM10"].shift(-1)
df.rename(columns={'Miesiąc': 'Month',
                     'Maksymalna temperatura dobowa': 'Max daily temp',
                     'Minimalna temperatura dobowa': 'Min daily temp',
                     'Średnia temperatura dobowa': 'Avg daily temp',
                     'Temperatura minimalna przy gruncie': 'Min ground temp',
                     'Suma dobowa opadu': 'Sum of daily fall',
                     'Wysokość pokrywy śnieżnej': 'Snow height',
                     'Równoważnik wodny śniegu': 'Water snow equivalent',
                     'Usłonecznienie': 'Insolation',
                     'Czas trwania opadu deszczu': 'Rain fall length',
                     'Czas trwania opadu śniegu': "Snow fall length",
                     'Czas trwania opadu deszczu ze śniegiem': 'RainSnow fall length',
                     'Czas trwania gradu': 'Hail fall length',
                     'Czas trwania mgły': 'Mist duration',
                     'Czas trwania zamglenia': 'Misty duration',
                     'Czas trwania sadzi': 'Rime frost duration',
                     'Czas trwania gołoledzi': 'Glaze duration',
                     'Czas trwania zamieci śnieżnej niskiej': 'Low blizzard duration',
                     'Czas trwania zamieci śnieżnej wysokiej': 'High blizzard duration',
                     'Czas trwania zmętnienia': 'Turbidity duration',
                     'Czas trwania wiatru >=10m/s': 'Wind 10more duration',
                     'Czas trwania wiatru >15m/s': 'Wind 15more duration',
                     'Czas trwania burzy': 'Storm duration',
                     'Czas trwania rosy': 'Dew duartion',
                     'Czas trwania szronu': 'Hoarfrost duration',
                     'Wystąpienie pokrywy śnieżnej': 'Snow cover',
                     'Izoterma dolna': 'Low isotherm',
                     'Izoterma górna': 'High isotherm',
                     'Aktynometria': 'Acrinometry',
                     'Średnie dobowe zachmurzenie ogólne': 'Avg daily general clouds',
                     'Średnia dobowa prędkość wiatru': 'Avg daily wind speed',
                     'Średnia dobowe ciśnienie pary wodnej': 'Avg steam pressure',
                     'Średnia dobowa wilgotność względna': 'Avg daily humidity relative',
                     'Średnia dobowe ciśnienie na poziomie stacji': 'Avg daily pressure station level',
                     'Średnie dobowe ciśnienie na pozimie morza': 'Avg daily pressure sea level',
                     'Suma opadu dzień': 'Day fall',
                     'Suma opadu noc': 'Night fall'}, inplace=True)

for file in raw_files:
    raw_data_df = pd.read_excel(my_path + file, index_col=0)
    if file == "2016_PM10_1g.xlsx":
        raw_data_df = raw_data_df.iloc[5:]
    else:
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
df.to_csv("data_proper2.csv", sep=";")
