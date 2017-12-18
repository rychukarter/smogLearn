from urllib import request as urlreq
from urllib.error import URLError
import gzip
import json
import os
import pandas as pd
import datetime

years = list(range(2001, 2017))
months = list(range(1, 13))
warsaw_station = "375"


def get_url_list_kli():
    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/dobowe/klimat/"
    url_list = list()
    for y in years:
        for m in months:
            if m in range(1, 10):
                m = "0" + str(m)
            url_list.append(url + str(y) + "/" + str(y) + "_" + str(m) + "_k.zip")
    return url_list


def get_url_list_opad():
    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/dobowe/opad/"
    url_list = list()
    for y in years:
        for m in months:
            if m in range(1,10):
                m = "0" + str(m)
            url_list.append(url + str(y) + "/" + str(y) + "_" + str(m) + "_o.zip")
    return url_list


def get_url_list_synop():
    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/dobowe/synop/"
    url_list = list()
    for y in years:
        url_list.append(url + str(y) + "/" + str(y) + "_" + warsaw_station + "_s.zip")
    return url_list


def get_station_list():
    station_list = list()
    data = urlreq.urlopen("https://dane.imgw.pl/assets/libs/stacje.json")
    json_data = json.load(data)
    for x in json_data:
        if "WARSZAWA" in x["nazwa"]:
            station_list.append(str(x["kod"]))
    return station_list


def download_files(directory, url_list):
    for url in url_list:
        file = url.split("/", -1)[-1]
        file = directory + file
        print(file)
        try:
            urlreq.urlretrieve(url, file)
        except URLError as e:
            print(e)


def get_column_names(file):
    names = list()
    with open(file) as f:
        for line in f:
            names.append(line[:-1])
    return names


def get_file_list(directory):
    d_file_list = [directory + f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return d_file_list


def filter_weather_data(files):
    for f in files:
        sd_f = f.split("/", -1)[-1].split("_" + warsaw_station)[0]
        names_list = get_column_names("./format/" + sd_f + "_format.txt")
        df = pd.read_csv(f, delimiter=',', names=names_list, encoding="ISO-8859-1")
        df.index = df["Rok"].map(str) + "-" + df["Miesiąc"].map(str) + "-" + df["Dzień"].map(str)
        del df["Kod stacji"]
        del df["Nazwa stacji"]
        del df["Rok"]
        del df["Dzień"]
        for name in df.columns.values:
            if "Status" in name:
                del df[name]

        df.to_csv("./synop_out/new_" + f.split("/", -1)[-1], sep=";", encoding="UTF-8")


def filter_pollution_data(files):
    for f in files:
        df = pd.read_excel(f, index_col=0, delimiter=";")
        df = df["PM10"].iloc[1:]
        print(df)
        df.to_csv("./raw_data/daily/pollution_out2/" + f.split("/", -1)[-1][:-5] + ".csv")


def merge_csv(files, not_inc_header=False):
    with open("out_pollution4" + ".csv", "a") as f_out:
        header_saved = not_inc_header
        for file in files:
            with open(file) as f_in:
                header = next(f_in)
                if not header_saved:
                    f_out.write(header)
                    header_saved = False
                for line in f_in:
                    f_out.write(line)
