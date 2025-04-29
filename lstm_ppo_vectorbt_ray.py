# -*- coding: utf-8 -*-
"""
使用 gymnasium + Ray RLlib PPO (TensorFlow 2 + LSTM) 的交易示例脚本 (Python 3.11+)。
- Legacy API 栈 (支持 framework="tf2")  
- Trainer 使用 1 块 GPU，Worker 不占 GPU  
- 计算 RSI(14) & MACD 指标  
- 动作：0=Hold,1=Buy,2=Sell  
- 训练 20 轮后用 algo.save() 保存 checkpoint  
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf

# ——GPU 按需分配显存——
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class TradingEnv(gym.Env):
    """简易交易环境：RSI(14) + MACD 特征。"""
    metadata = {"render_modes": []}

    def __init__(self, env_config):
        super().__init__()
        path = env_config.get("csv_path", "btc.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到 CSV 文件: {path}")
        df = pd.read_csv(path)

        # RSI(14)
        delta = df["Close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        df["RSI"] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        df.dropna(subset=["RSI"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # MACD = EMA12 - EMA26
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26

        self.df = df
        self.n = len(df)

        # 动作 & 观测空间
        self.action_space = spaces.Discrete(3)
        low = np.array([0.0, 0.0, -np.inf, 0.0], dtype=np.float32)
        high = np.array([np.finfo(np.float32).max, 100.0, np.inf, 1.0],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.step_idx = 0
        self.position = 0
        self.buy_price = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.position = 0
        self.buy_price = 0.0
        return self._obs(), {}

    def step(self, action):
        price = self.df.loc[self.step_idx, "Close"]
        reward = 0.0
        if action == 1 and self.position == 0:
            self.position = 1
            self.buy_price = price
        elif action == 2 and self.position == 1:
            reward = price - self.buy_price
            self.position = 0
            self.buy_price = 0.0

        self.step_idx += 1
        terminated = self.step_idx >= self.n
        if terminated:
            self.step_idx = self.n - 1
        return self._obs(), reward, terminated, False, {}

    def _obs(self):
        row = self.df.loc[self.step_idx]
        return np.array(
            [row["Close"], row["RSI"], row["MACD"],
             float(self.position)],
            dtype=np.float32)


if __name__ == "__main__":
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    # 启动 Ray（自动检测 GPU）
    ray.init(ignore_reinit_error=True)

    # 构建 PPOConfig：Legacy API 栈 + TF2 + LSTM + GPU
    config = (PPOConfig().environment(
        env=TradingEnv, env_config={
            "csv_path": "btc.csv"
        }).framework("tf2").api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False).env_runners(
                num_env_runners=0).resources(
                    num_gpus=1,
                    num_gpus_per_worker=0).training(model={
                        "use_lstm": True,
                        "max_seq_len": 20,
                        "lstm_cell_size": 256
                    }))
    algo = config.build()

    # 训练 20 轮，打印平均回报
    for i in range(1, 21):
        result = algo.train()
        runners = result["env_runners"]
        mean_r = runners.get("episode_reward_mean",
                             runners.get("episode_return_mean", float("nan")))
        print(f"Iteration {i:02d} → mean reward: {mean_r:.2f}")

    # ——这里改为 save() ——
    ckpt = algo.save(
    )  # ⚙️ 使用 save()，替代不支持旧栈的 save_to_path() :contentReference[oaicite:2]{index=2}
    print(f"Checkpoint saved at: {ckpt}")

    ray.shutdown()
