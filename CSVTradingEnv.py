import gymnasium as gym
from gymnasium import spaces
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import ta


class CSVTradingEnv(gym.Env):

    def __init__(self, csv_file, window_size=1, max_steps=None):
        super().__init__()

        self.data = pd.read_csv(csv_file)
        self.data["rsi"] = ta.momentum.RSIIndicator(self.data["Close"]).rsi()
        self.data["macd"] = ta.trend.MACD(self.data["Close"]).macd_diff()
        self.data.dropna(inplace=True)
        self.close_prices = self.data["Close"].values
        self.rsi = self.data["rsi"].values
        self.macd = self.data["macd"].values

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(3, ),
                                            dtype=np.float32)

        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

    def _get_obs(self):
        return np.array([
            self.close_prices[self.step_count], self.rsi[self.step_count],
            self.macd[self.step_count]
        ],
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.position = 0
        self.current_price = self.close_prices[0]
        obs = np.array([self.current_price, self.position], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps - 1

        prev_price = self.close_prices[self.step_count - 1]
        self.current_price = self.close_prices[self.step_count]

        reward = 0
        if action == 1:  # buy
            self.position = 1
        elif action == 2:  # sell
            if self.position == 1:
                reward = self.current_price - prev_price
            self.position = 0
        elif action == 0 and self.position == 1:
            reward = self.current_price - prev_price

        obs = np.array([self.current_price, self.position], dtype=np.float32)
        return obs, reward, done, False, {}

    def render(self):
        print(
            f"Step {self.step_count}: Price = {self.current_price:.2f}, Position = {self.position}"
        )
