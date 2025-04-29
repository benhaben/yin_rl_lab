import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import ta

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import get_device

import backtrader as bt

def compute_indicators(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd_diff()
    df.dropna(inplace=True)
    return df

class TradingEnv(gym.Env):
    """Gym environment with RSI and MACD features."""
    def __init__(self, csv_file, max_steps=200):
        super().__init__()
        df = pd.read_csv(csv_file)
        df = df.dropna().reset_index(drop=True)
        df = compute_indicators(df)

        self.df = df.reset_index(drop=True)
        self.close = self.df['Close'].values
n        self.rsi = self.df['rsi'].values
        self.macd = self.df['macd'].values
        self.max_steps = min(max_steps, len(self.df)-1)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=hold,1=buy,2=sell

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return np.array([
            self.close[self.current_step],
            self.rsi[self.current_step],
            self.macd[self.current_step]
        ], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        done = False
        price = self.close[self.current_step]
        self.current_step += 1
        # actions
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 1:
            reward = price - self.entry_price
            self.position = 0
        elif action == 0 and self.position == 1:
            reward = price - self.entry_price

        done = self.current_step >= self.max_steps
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def render(self):
        print(f"Step {self.current_step}, Price: {self.close[self.current_step]:.2f}, Position: {self.position}")

# —— Training ——

def train_model():
    env = DummyVecEnv([lambda: TradingEnv('btc.csv')])
    model = RecurrentPPO('MlpLstmPolicy', env, verbose=1, device=get_device())
    model.learn(total_timesteps=20000)
    model.save('lstm_trading_model')
    return model

# —— Backtrader 策略 ——
class RLStrategy(bt.Strategy):
    params = dict(model_path='lstm_trading_model', csv_file='btc.csv')

    def __init__(self):
        # load SB3 model
        self.model = RecurrentPPO.load(self.p.model_path)
        self.lstm_state = None
        self.episode_start = True
        # data arrays
        self.dataclose = self.datas[0].close
        self.rsi = bt.indicators.RSI(self.datas[0], period=14)
        macd = bt.indicators.MACD(self.datas[0])
        self.macd = macd.macd - macd.signal

    def next(self):
        obs = np.array([
            self.dataclose[0],
            self.rsi[0],
            self.macd[0]
        ], dtype=np.float32)
        action, self.lstm_state = self.model.predict(obs, state=self.lstm_state, episode_start=np.array([self.episode_start]))
        self.episode_start = False

        if action == 1 and not self.position:
            self.buy()
        elif action == 2 and self.position:
            self.sell()

# —— Backtesting ——

def backtest():
    cerebro = bt.Cerebro()
    df = pd.read_csv('btc.csv', parse_dates=True, index_col='Date')
    data = bt.feeds.PandasData(dataname=compute_indicators(df))
    cerebro.adddata(data)
    cerebro.addstrategy(RLStrategy)
    cerebro.broker.setcash(10000)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)

    print('Starting Portfolio Value:', cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value:', cerebro.broker.getvalue())
    cerebro.plot()

if __name__ == '__main__':
    train_model()
    backtest()
