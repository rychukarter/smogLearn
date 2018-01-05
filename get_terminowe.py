from daily import daily

url_list = daily.get_url_list_termin_synop()
daily.download_files("./raw_data/daily/terminowe/", url_list)
