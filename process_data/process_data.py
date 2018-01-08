from urllib import request as urlreq
from urllib.error import URLError
import json
import os
import pandas as pd
import pandas.errors as pderr

#years = list(range(2001, 2017))
#months = list(range(1, 13))
warsaw_station = "375"


def generate_url_list_kli_dob(years, months):
    """Generates list of urls from IMGW's dobowe/klimat directory.

    Since klimat data is a subset of synop data this function is not used.

    :param years: numeric list of years user is interested in
    :param months: numeric list of months user is interested in. It is kept same for every year in yeras list.
    :return: list of generated urls used to download files
    """

    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/dobowe/klimat/"
    url_list = list()
    for y in years:
        for m in months:
            if m in range(1, 10):
                m = "0" + str(m)
            url_list.append(url + str(y) + "/" + str(y) + "_" + str(m) + "_k.zip")

    return url_list


def generate_url_list_opad_dob(years, months):
    """Generates list of urls from IMGW's dobowe/opad directory.

    Since opad data is a subset of synop data this function is not necessarely  used.

    :param years: numeric list of years user is interested in
    :param months: numeric list of months user is interested in - kept the same for every year in years list
    :return: list of generated urls used to download files
    """

    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/dobowe/opad/"
    url_list = list()
    for y in years:
        for m in months:
            if m in range(1, 10):
                m = "0" + str(m)
            url_list.append(url + str(y) + "/" + str(y) + "_" + str(m) + "_o.zip")

    return url_list


def generate_url_list_synop_dob(years, station=warsaw_station):
    """Generates list of urls from IMGW's dobowe/synop directory.

    :param years: numeric list of years user is interested in
    :param station: last 3 numbers of ID of station user is interested in (default warsaw_station)
    :return: list of generated urls used to download files
    """

    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/dobowe/synop/"
    url_list = list()
    for y in years:
        url_list.append(url + str(y) + "/" + str(y) + "_" + station + "_s.zip")

    return url_list


def generate_url_list_kli_term(years, months):
    """Generates list of urls from IMGW's terminowe/klimat directory.

    Since klimat data is a subset of synop data this function is not necessarily used.

    :param years: numeric list of years user is interested in
    :param months: numeric list of months user is interested in. It is kept same for every year in yeras list.
    :return: list of generated urls used to download files
    """

    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/klimat/"
    url_list = list()
    for y in years:
        for m in months:
            if m in range(1, 10):
                m = "0" + str(m)
            url_list.append(url + str(y) + "/" + str(y) + "_" + str(m) + "_k.zip")

    return url_list


def generate_url_list_opad_term(years, months):
    """Generates list of urls from IMGW's terminowe/opad directory.

    Since opad data is a subset of synop data this function is not necessarily used.

    :param years: numeric list of years user is interested in
    :param months: numeric list of months user is interested in - kept the same for every year in years list
    :return: list of generated urls used to download files
    """

    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/opad/"
    url_list = list()
    for y in years:
        for m in months:
            if m in range(1, 10):
                m = "0" + str(m)
            url_list.append(url + str(y) + "/" + str(y) + "_" + str(m) + "_o.zip")

    return url_list


def generate_url_list_synop_term(years, station=warsaw_station):
    """Generates list of urls from IMGW's terminowe/synop directory.

    :param years: numeric list of years user is interested in
    :param station: last 3 numbers of ID of station user is interested in (default warsaw_station)
    :return: list of generated urls used to download files
    """

    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/"
    url_list = list()
    for y in years:
        url_list.append(url + str(y) + "/" + str(y) + "_" + station + "_s.zip")

    return url_list


def get_stations_list():
    """Gets all IMGW's measurement station IDs.

    :return: list of stations IDs
    """

    station_list = list()
    json_data = json.load(urlreq.urlopen("https://dane.imgw.pl/assets/libs/stacje.json"))
    for x in json_data:
        station_list.append(str(x["kod"]))

    return station_list


def get_stations_list_warsaw():
    """Gets Warsaw IMGW's measurement station IDs.

    :return: list of Warsaw stations IDs
    """

    station_list = list()
    json_data = json.load(urlreq.urlopen("https://dane.imgw.pl/assets/libs/stacje.json"))
    for x in json_data:
        if "WARSZAWA" in x["nazwa"]:
            station_list.append(str(x["kod"]))

    return station_list


def download_files(directory, url_list):
    """Downloads files.

    Function to download all files from url_list. If file is missing, it is skipped and exception is thrown.

    :param directory: location to save files in
    :param url_list: list of urls to download
    :return:
    """

    for url in url_list:
        file = directory + url.split("/", -1)[-1]
        try:
            urlreq.urlretrieve(url, file)
        except URLError as e:
            print(e)


def get_column_names(format_file):
    """Gets columns names for IMGW's data from format file

    :param format_file: format file to be used form ./raw_data/daily/format/ directory
    :return: list of column names
    """

    names = list()
    with open(format_file) as f:
        for line in f:
            names.append(line[:-1])

    return names


def get_file_list(directory):
    """Gets paths to files in directory.

    :param directory: path to directory to look up
    :return: list of paths to files in directory
    """

    d_file_list = [directory + f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return d_file_list


def merge_weather_csv(files, out_file, include_header=False):
    """Merges csv weather data files into one in their name order.

    :param files: files list to be merged
    :param out_file: path to output file
    :param include_header: defines if files have headers - if so uses first line of first file as output file header
    :return:
    """

    with open(out_file, "a") as f_out:
        inc_header = include_header
        header_saved = False

        for file in files:
            with open(file) as f_in:
                if inc_header:
                    header = next(f_in)
                    if not header_saved:
                        f_out.write(header)
                        header_saved = True
                for line in f_in:
                    f_out.write(line)


def get_pollution_24h_data(files, column_name):
    """Gets date-indexed pollution 24h data from files based on column name (station).

    As there is no consistency in naming selected column should be renamed in all files to same value manually.
    Also format has changed since 2016, so if this file is used is should be cut differently.

    :param files: list of files with pollution data
    :param column_name: name of column to get data from
    :return: dataframe with pollution 24h data in chronological order
    """

    frames = list()
    out_df = pd.DataFrame()
    for f in files:
        print("Getting daily pollution from:", f)
        df = pd.read_excel(f, index_col=0)
        if "2016" in f:
            df[column_name] = df[column_name].str.replace(',', '.')
            frames.append(df[column_name].iloc[0:])
        else:
            frames.append(df[column_name].iloc[2:])

    out_df = pd.concat(frames)
    return out_df


def get_pollution_1h_data(files, column_name):
    """Gets dat-indexed pollution 1h data from files based on column name (station).

    As there is no consistency in naming selected column should be renamed in all files to same value manually.
    Also format has changed since 2016, so if this file is used is should be cut differently. Data is transformed
    in a way that every hour becomes separate column. Each hour column name is created by adding "_hour" to
    original name.

    :param files: list of files with pollution data
    :param column_name: name of column to get data from
    :return: dataframe with pollution 1h data in chronological order
    """

    out_df = pd.DataFrame()

    for f in files:
        print("Getting hourly pollution from:", f)
        raw_data_df = pd.read_excel(f, index_col=0)
        if "2016" in f:
            raw_data_df = raw_data_df.iloc[5:]
            raw_data_df[column_name] = raw_data_df[column_name].str.replace(',', '.')
        else:
            raw_data_df = raw_data_df.iloc[3:]
        raw_data_df.index = pd.to_datetime(raw_data_df.index)
        for index, row in raw_data_df.iterrows():
            try:
                out_df.at[index.date(), "PM10_h_" + str(index.hour)] = row[column_name]
            except Exception as e:
                print(e)

    return out_df


def get_processed_weather_data_dob(files, format_dir, station=warsaw_station):
    """Gets date-indexed daily weather data based on station.

    Since synop data is splitted into s_d and s_d_t subsets with different columns it is necessary to treat
    them differently. "Datetime-able" index is created based on time columns. Obsolete columns are deleted.

    :param files: synop raw data files list
    :param format_dir: path to directory with formats
    :param station: last 3 numbers of ID of station user is interested in (default warsaw_station)
    :return: dataframe with daily weather data
    """

    frames_s_d = list()
    frames_s_d_t = list()
    for f in files:
        print("Getting daily weather from:", f)
        names_list = get_column_names(format_dir + f.split("/", -1)[-1].split("_" + station)[0]
                                      + "_format.txt")
        df = pd.read_csv(f, delimiter=',', names=names_list, encoding="ISO-8859-1")
        df.index = df["Rok"].map(str) + "-" + df["Miesiąc"].map(str) + "-" + df["Dzień"].map(str)
        df.index = pd.to_datetime(df.index)
        for name in df.columns.values:
            for x in ["Kod stacji", "Nazwa stacji", "Rok", "Dzień", "Stan gruntu Z/R", "Rodzaj opadu",
                      "Wystąpienie błyskawicy", "Średnia dobowa temperatura", "Status"]:
                if x in name:
                    del df[name]
        if "s_d_t" in f:
            frames_s_d_t.append(df)
        else:
            frames_s_d.append(df)

    return pd.concat([pd.concat(frames_s_d), pd.concat(frames_s_d_t)], axis=1)


def get_processed_weather_data_term(files, format_dir, columns, station=warsaw_station):
    """Gets date-indexed hourly weather data based on selected columns and station.

    Function splits every hour to be separate column in returned dataframe for every provided column name. Each
    hour column name is created by adding "_hour" to original name.

    :param files: terminowe raw data files list
    :param format_dir: path to directory with formats
    :param columns: columns to be returned
    :param station: last 3 numbers of ID of station user is interested in (default warsaw_station)
    :return: dataframe with hourly weather data
    """

    out_df = pd.DataFrame()

    for f in files:
        print("Getting hourly weather from:", f)
        names_list = get_column_names(format_dir + f.split("/", -1)[-1].split("_" + station)[0]
                                      + "_format.txt")
        df = pd.read_csv(f, delimiter=',', names=names_list, encoding="ISO-8859-1")
        df.index = df["Rok"].map(str) + "-" + df["Miesiąc"].map(str) + "-" + df["Dzień"].map(str)
        df.index = pd.to_datetime(df.index)
        for c in columns:
            for index, row in df.iterrows():
                try:
                    out_df.at[index.date(), c + "_" + str(row["Godzina"])] = row[c]
                except Exception as e:
                    print(e)
    return out_df


