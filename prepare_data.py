from process_data import process_data
import pandas as pd
from os import listdir
from os.path import isfile, join

synop_dir = "./raw_data/daily/synop/"
terminowe_dir = "./raw_data/daily/terminowe/"
pollution_24h_dir = "./raw_data/daily/pollution/"
pollution_1h_dir = "./raw_data/daily/pollution_1g/"
format_dir = "./raw_data/daily/format/"

synop_files = process_data.get_file_list(synop_dir)
terminowe_files = process_data.get_file_list(terminowe_dir)
pollution_24_files = process_data.get_file_list(pollution_24h_dir)
pollution_1h_files = process_data.get_file_list(pollution_1h_dir)

pollution_1h_data = process_data.get_pollution_1h_data(pollution_1h_files, "HERE")
print("\n----------------------------------------------------------\n")
feature_list = ["Kierunek wiatru", "Prędkość wiatru", "Temperatura powietrza", "Ciśnienie na pozimie stacji",
                "Wartość tendencji", "Charakterystyka tendencji"]
weather_1h_data = process_data.get_processed_weather_data_term(terminowe_files,format_dir, feature_list)
print("\n----------------------------------------------------------\n")
pollution_24h_data = process_data.get_pollution_24h_data(pollution_24_files, "PM10")
print("\n----------------------------------------------------------\n")
weather_24h_data = process_data.get_processed_weather_data_dob(synop_files, format_dir)
print("\n----------------------------------------------------------\n")

df = pd.concat([weather_24h_data, weather_1h_data, pollution_24h_data, pollution_1h_data], axis=1)

df["PM10_prev"] = df["PM10"].shift(1)
df["PM10_next"] = df["PM10"].shift(-1)

df = df.dropna(axis=0, how="any")
df.to_csv("out.csv", sep=";")
print(df.dtypes)


'''
print("\n----------------------------------------------------------\n")
df.rename(columns={'Miesiąc': 'Month',
                   'Maksymalna temperatura dobowa': 'Max process_data temp',
                   'Minimalna temperatura dobowa': 'Min process_data temp',
                   'Średnia temperatura dobowa': 'Avg process_data temp',
                   'Temperatura minimalna przy gruncie': 'Min ground temp',
                   'Suma dobowa opadu': 'Sum of process_data fall',
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
                   'Średnie dobowe zachmurzenie ogólne': 'Avg process_data general clouds',
                   'Średnia dobowa prędkość wiatru': 'Avg process_data wind speed',
                   'Średnia dobowe ciśnienie pary wodnej': 'Avg steam pressure',
                   'Średnia dobowa wilgotność względna': 'Avg process_data humidity relative',
                   'Średnia dobowe ciśnienie na poziomie stacji': 'Avg process_data pressure station level',
                   'Średnie dobowe ciśnienie na pozimie morza': 'Avg process_data pressure sea level',
                   'Suma opadu dzień': 'Day fall',
                   'Suma opadu noc': 'Night fall'}, inplace=True)
'''
