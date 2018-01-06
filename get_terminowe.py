from daily import daily
import pandas as pd
from os import listdir
from os.path import isfile, join

#url_list = daily.get_url_list_termin_synop()
#daily.download_files("./raw_data/daily/terminowe/", url_list)

#my_path = "./raw_data/daily/terminowe/"
#files = [join(my_path, f) for f in listdir(my_path) if isfile(join(my_path, f))]
#daily.merge_csv(files, "./raw_data/daily/terminowe_out/out.csv")
#daily.filter_weather_data(files, "./raw_data/daily/terminowe_out/")

names = daily.get_column_names("./raw_data/daily/format/s_t_format.txt")
print(names)
df = pd.read_csv("./raw_data/daily/terminowe_out/out.csv", delimiter=",", names=names)
df.index = df["Rok"].map(str) + "-" + df["Miesiąc"].map(str) + "-" + df["Dzień"].map(str)
df.to_csv("test.csv", sep=";")
