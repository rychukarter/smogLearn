from daily import daily as day
from hourly import hourly as hour




# df = pd.read_csv("data2.csv", delimiter=";", index_col=0)
# del df["Stan gruntu Z/R"]
# del df["Rodzaj opadu"]
# df = df.dropna(axis=0, how="any")
# empty_rows = df.isnull().sum()
# print(empty_rows)
# print(df)

# df.to_csv("data2.csv", sep=";")

li = day.get_url_list_kli()
hi = hour.months
print(hi)
