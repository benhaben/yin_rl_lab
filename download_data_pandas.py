import pandas_datareader.data as web
import datetime

start = datetime.datetime(2022, 1, 1)
end = datetime.datetime(2023, 1, 1)

btc = web.DataReader("BTC-USD", "yahoo", start, end)
btc.to_csv("btc.csv")
