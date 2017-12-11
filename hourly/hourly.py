from urllib import request as urlreq
from urllib.error import URLError
import gzip
import json
import os
import pandas as pd

years = list(range(2007, 2018))
months = list(range(1, 13))


def get_file_list(url_list):
    data = urlreq.urlopen("https://dane.imgw.pl/assets/libs/stacje.json")
    json_data = json.load(data)
    for y in years:
        for m in months:
            if m in range(1, 10):
                m = "0" + str(m)
            for x in json_data:
                if "WARSZAWA" in x["nazwa"]:
                    url_list.append("https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/" + str(y) + "/" + str(m) +
                                    "/dane_" + str(y) + "_" + str(m) + "_" + str(x["kod"]) + ".csv.gz")


def get_code_list(code_list):
    data = urlreq.urlopen("https://dane.imgw.pl/assets/libs/klasyfikacje.json")
    json_data = json.load(data)
    for x in json_data:
        if "temperatura" in x["nazwa"].lower():
            code_list.append(x["kod"])
        elif "ciśnienie" in x["nazwa"].lower():
            code_list.append(x["kod"])
        elif "opad" in x["nazwa"].lower():
            code_list.append(x["kod"])
        elif "wilgotno" in x["nazwa"].lower():
            code_list.append(x["kod"])
        elif "wiatr" in x["nazwa"].lower():
            code_list.append(x["kod"])


def download_files(url_list):
    for url in url_list:
        file = url.split("/", -1)[-1]
        print(url, file)
        try:
            urlreq.urlretrieve(url, file)
        except URLError as e:
            print(e)


def extract_files():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        if ".csv.gz" in f:
            zip_file = gzip.open(f, 'rb')
            print("Extracting:", zip_file.name)
            out_file = open(zip_file.name[:-7] + ".csv", 'wb')
            out_file.write(zip_file.read())
            zip_file.close()
            out_file.close()


def panda_way(files, codes):
    for file in files:
        df = pd.read_csv("../raw_data/" + file, delimiter=";", names=['number', 'date', 'code', 'value'])
        df = df.pivot(index='date', columns='code', values='value')
        for x in df.columns.values:
            if x not in codes:
                del df[x]
        df.index = pd.to_datetime(df.index)
        df = df.ix[df.index.minute == 00]

        empty_rows = df.isnull().sum()
        full_rows = empty_rows.loc[empty_rows == 0]
        df = df[full_rows.index]
        if df.empty:
            print("DROPED FILE:", file)
            continue
        print("FILE:", file)
        df.to_csv(file, sep=';')


def merge_csv(files):
    current_file = files[0]
    while current_file:
        print("CURRENT FILE:", current_file)
        f_out = open(current_file[-13:], "a")
        header_saved = False
        remove_list = list()
        for file in files:
            if file[-13:] == current_file[-13:]:
                with open(file) as f_in:
                    print("\t\tCoping file:", file)
                    header = next(f_in)
                    if not header_saved:
                        f_out.write(header)
                        header_saved = True
                    for line in f_in:
                        f_out.write(line)

                remove_list.append(file)
        files = [x for x in files if x not in remove_list]
        #os.remove(remove_list)
        f_out.close()
        if files:
            current_file = files[0]
        else:
            break
