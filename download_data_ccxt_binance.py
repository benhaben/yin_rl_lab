import ccxt
import pandas as pd
import time
from datetime import datetime
import os

import requests

proxies = {
    "http": "http://127.0.0.1:7890",  # Clash 默认端口
    "https": "http://127.0.0.1:7890",
}

response = requests.get("https://api.ipify.org?format=json", proxies=proxies)
print(response.json())

response1 = requests.get("https://api.binance.com", proxies=proxies)
print(response1)

# 可选：使用 API Key 提高速率
API_KEY = "2CQKaRtJFBV1hqri2fz8lI8UPhTZhJBYkVfTHvmq1XdHpWq0X3lRPDZi1R0iINpV"
API_SECRET = "huRkLFARLquMdIZQYz4eLDn8gzTyR5a3oePHC8qP2NPekIvzAhN03O5TPjNB1mDg"


def fetch_ohlcv(symbol="BTC/USDT",
                timeframe="1d",
                years=2,
                limit=500,
                save_interval=50000):
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'proxies': proxies,
    })
    exchange.enableRateLimit = True  # 自动管理速率

    # 计算起始时间（毫秒）
    end_time = exchange.milliseconds()
    start_time = end_time - years * 365 * 24 * 60 * 60 * 1000

    # 格式化时间
    start_date = pd.to_datetime(start_time, unit='ms').strftime('%Y%m%d')
    end_date = pd.to_datetime(end_time, unit='ms').strftime('%Y%m%d')

    # 生成文件名
    filename = f"{symbol.replace('/', '_')}_{timeframe}_{start_date}_{end_date}.parquet"
    file_path = os.path.join(os.getcwd(), filename)

    all_data = []

    # 断点续传
    if os.path.exists(file_path):
        df_existing = pd.read_parquet(file_path)
        start_time = int(df_existing['timestamp'].max().timestamp() * 1000) + 1
        all_data.extend(df_existing.values.tolist())
        print(f"Resuming from {df_existing['timestamp'].max()}...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, start_time, limit)
            if not ohlcv:
                break

            all_data.extend(ohlcv)
            start_time = ohlcv[-1][0] + 1  # 避免重复

            print(f"Fetched {len(all_data)} records so far...")

            # 每 50,000 条数据存储一次
            if len(all_data) % save_interval == 0:
                df = pd.DataFrame(all_data,
                                  columns=[
                                      'timestamp', 'open', 'high', 'low',
                                      'close', 'volume'
                                  ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.to_parquet(file_path, index=False)
                print(f"Saved {len(all_data)} records to {file_path}")

            # 达到目标数据量
            if len(all_data) >= years * 365 * 288:
                break

            time.sleep(3)  # 降低请求频率，避免超时

        except ccxt.NetworkError as e:
            print(f"Network Error: {e}, retrying in 10s...")
            time.sleep(10)  # 避免短时间 API 失败导致崩溃

        except ccxt.ExchangeError as e:
            print(f"Exchange Error: {e}, retrying in 30s...")
            time.sleep(30)

        except Exception as e:
            print(f"Unexpected Error: {e}, retrying in 60s...")
            time.sleep(60)

    df = pd.DataFrame(
        all_data,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")

    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


# 设置时间范围
start_date = "2022-01-01"
since_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)

df = fetch_ohlcv(symbol="ETH/USDT", years=8)
df.to_csv("eth.csv", index=False)
print("Saved to eth.csv")
