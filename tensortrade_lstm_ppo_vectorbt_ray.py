import pandas as pd
import numpy as np
import gym
import ta  # 技术指标库
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from tensortrade.env.default import create
from tensortrade.feed.core import DataFeed, Stream
from gym.spaces import Discrete, Box


# ---------- Step 1: Load and process BTC data ----------
def load_btc_data():
    df = pd.read_csv("btc.csv", parse_dates=["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    # 技术指标：RSI + MACD
    df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
    macd = ta.trend.MACD(close=df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df = df[["Close", "rsi", "macd", "macd_signal"]]
    df = df.fillna(method="bfill").fillna(method="ffill")

    return df


df = load_btc_data()


# ---------- Step 2: Create Gym-compatible custom environment ----------
class BTCEnv(gym.Env):

    def __init__(self, config=None):
        self.df = df.copy()
        self.window_size = 20
        self.frame_bound = (self.window_size, len(self.df))
        self.current_step = self.frame_bound[0]
        self.action_space = Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = Box(low=-np.inf,
                                     high=np.inf,
                                     shape=(self.window_size, 4),
                                     dtype=np.float32)
        self.position = 0  # 0: neutral, 1: long, -1: short
        self.entry_price = 0
        self.total_profit = 0

    def reset(self):
        self.current_step = self.frame_bound[0]
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        return self._get_observation()

    def _get_observation(self):
        window = self.df.iloc[self.current_step -
                              self.window_size:self.current_step]
        return window.values.astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["Close"]

        reward = 0

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
        elif action == 1 and self.position == -1:
            reward = self.entry_price - current_price
            self.total_profit += reward
            self.position = 0
        elif action == 2 and self.position == 1:
            reward = current_price - self.entry_price
            self.total_profit += reward
            self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        obs = self._get_observation()
        info = {"total_profit": self.total_profit}
        return obs, reward, done, info


# ---------- Step 3: Register env and train with Ray ----------
ray.shutdown()
ray.init(ignore_reinit_error=True)

tune.register_env("btc_env", lambda config: BTCEnv(config))

analysis = tune.run(PPOTrainer,
                    config={
                        "env": "btc_env",
                        "framework": "torch",
                        "num_workers": 1,
                        "model": {
                            "use_lstm": True,
                            "lstm_cell_size": 64,
                        },
                        "env_config": {},
                    },
                    stop={"training_iteration": 20},
                    verbose=1)

# ---------- Optional: print best result ----------
print("Best result:", analysis.best_result["episode_reward_mean"])
