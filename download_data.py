import yfinance as yf

data = yf.download("BTC-USD",
                   start="2022-01-01",
                   end="2023-01-01",
                   interval="1d",
                   progress=False)
data.to_csv("btc.csv")
