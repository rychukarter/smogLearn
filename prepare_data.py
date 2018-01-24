from process_data import process_data
import pandas as pd

synop_dir = "./raw_data/daily/synop/"
terminowe_dir = "./raw_data/daily/terminowe/"
pollution_24h_dir = "./raw_data/daily/pollution/"
pollution_1h_dir = "./raw_data/daily/pollution_1g/"
format_dir = "./raw_data/daily/format/"
feature_list = ["Widzialność", "Widzialność operatora", "Kierunek wiatru", "Prędkość wiatru", "Temperatura powietrza",
                "Wilgotność względna", "Ciśnienie pary wodnej", "Ciśnienie na pozimie stacji",
                "Ciśnienie na pozimie morza", "Charakterystyka tendencji", "Wartość tendencji",
                "Temperatura punktu rosy"]

synop_files = process_data.get_file_list(synop_dir)
terminowe_files = process_data.get_file_list(terminowe_dir)
pollution_24_files = process_data.get_file_list(pollution_24h_dir)
pollution_1h_files = process_data.get_file_list(pollution_1h_dir)

pollution_1h_data = process_data.get_pollution_1h_data(pollution_1h_files, "HERE")
print("\n----------------------------------------------------------\n")
weather_1h_data = process_data.get_processed_weather_data_term(terminowe_files, format_dir, feature_list)
print("\n----------------------------------------------------------\n")
pollution_24h_data = process_data.get_pollution_24h_data(pollution_24_files, "PM10")
print("\n----------------------------------------------------------\n")
weather_24h_data = process_data.get_processed_weather_data_dob(synop_files, format_dir)
print("\n----------------------------------------------------------\n")

df = pd.concat([weather_24h_data, pollution_24h_data], axis=1)
df.index = pd.to_datetime(df.index)
df["Dzień tygodnia"] = df.index.weekday
df["PM10_prev"] = df["PM10"].shift(1)
df["PM10_next"] = df["PM10"].shift(-1)
df = df.dropna(axis=0, how="any")
df.to_csv("data_daily.csv", sep=";")

df2 = pd.concat([weather_24h_data, weather_1h_data, pollution_24h_data, pollution_1h_data], axis=1)
df2.index = pd.to_datetime(df2.index)
df2["Dzień tygodnia"] = df2.index.weekday
df2["PM10_prev"] = df2["PM10"].shift(1)
df2["PM10_next"] = df2["PM10"].shift(-1)
df2 = df2.dropna(axis=0, how="any")
df2.to_csv("data_daily_hourly.csv", sep=";")


