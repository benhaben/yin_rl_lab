# train_btc_sb3.py
# -*- coding: utf-8 -*-
"""
使用 Gymnasium + Stable-Baselines3 (SB3) RecurrentPPO (LSTM) 训练 BTC 交易策略示例。
- 读取 btc.csv，计算 RSI(14) & MACD 特征
- 自定义 Gymnasium 环境，动作：0=Hold，1=Buy，2=Sell
- 使用 sb3_contrib.RecurrentPPO 的 MlpLstmPolicy 训练 LSTM 策略
- GPU 加速（若安装了 GPU 版 PyTorch）
- 保存并演示模型预测
关键点对比
环境定义：与 Ray 版本几乎相同，保留 RSI & MACD 特征和三动作逻辑。

模型创建：

Ray RLlib：需要配置 PPOConfig、.api_stack()，并调用 .build()→.train()；

SB3：一句 RecurrentPPO(…)，自动处理策略、Rollout、优化等，学习过程由 model.learn() 驱动，更加简洁。

LSTM 支持：

Ray：需手动开启 use_lstm、lstm_cell_size；

SB3：直接选用 MlpLstmPolicy，底层自动集成 LSTM。

GPU 加速：

Ray：在配置中指定 GPU 资源；

SB3：通过 device="cuda" 让 PyTorch 自动使用可见 GPU。

日志 & 可视化：

SB3 直接支持 TensorBoard（tensorboard_log），可在训练时实时监控；

Ray 则需额外启动 ray dashboard 或配置 Tune 日志回调。

这样，SB3 的整体代码量更少，API 更集中，如果你更倾向快速上手和简洁易用，SB3 确实更轻量


SB3（Stable-Baselines3）**完全基于 PyTorch** 实现，官方文档明确指出它是 “a set of reliable implementations of reinforcement learning algorithms in PyTorch”([Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/?utm_source=chatgpt.com))([GitHub](https://github.com/DLR-RM/stable-baselines3?utm_source=chatgpt.com))。因此，如果你当前环境只安装了 TensorFlow（即使是 GPU 版），**SB3 本身是无法运行的**，因为它内部所有模型、张量操作、优化器都依赖 PyTorch 库。

---

## 选项一：安装 PyTorch 以继续使用 SB3

1. 创建或激活你的虚拟环境（推荐 Python 3.9–3.11）：  
   ```bash
   conda activate your_env  # or source venv/bin/activate
   ```
2. 安装支持 GPU 的 PyTorch：  
   ```bash
   # 以 CUDA 12.8 为例，请根据 https://pytorch.org 获取对应命令
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. 安装 SB3（此时会自动检测到 PyTorch）：  
   ```bash
   pip install stable-baselines3
   ```
4. 重新运行你的 `train_btc_sb3.py` 脚本，GPU 应可被 PyTorch 利用。

> **SB3 安装要求**：  
> - Python ≥ 3.9  
> - PyTorch ≥ 1.9（或 ≥ 2.0 更佳）  
> - 可选：`sb3_contrib` 用于 RecurrentPPO 等实验性算法  
>  ([Installation — Stable Baselines3 2.6.1a0 documentation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html?utm_source=chatgpt.com))  

---

## 选项二：使用纯 TensorFlow 的 RL 库

如果你希望**仅靠 TensorFlow**（无需安装 PyTorch），可以考虑以下几个主流库：

1. **TF-Agents**  
   - 官方 TensorFlow RL 库，直接兼容 TensorFlow 2.x（CPU/GPU）  
   - 安装：  
     ```bash
     pip install tf-agents
     ```  
   - 文档与教程：TF-Agents 一套核心概念（Agents/Policies/Environments）完全基于 TensorFlow 张量与 `tf.function`，可在 GPU 上高效运行。 ([tf-agents · PyPI](https://pypi.org/project/tf-agents/?utm_source=chatgpt.com), [GitHub - tensorflow/agents: TF-Agents: A reliable, scalable and easy to use TensorFlow library for Contextual Bandits and Reinforcement Learning.](https://github.com/tensorflow/agents?utm_source=chatgpt.com))  
2. **Keras-RL2**  
   - 基于 Keras API 的 RL 库，支持 DQN、DDPG、SAC 等多种算法  
   - 安装：  
     ```bash
     pip install keras-rl2
     ```  
3. **TensorLayer RL**  
   - 基于 TensorLayer 构建，面向研究和教学  
   - 安装：  
     ```bash
     pip install tensorlayer
     ```  

这些库都可以在**纯 TensorFlow GPU** 环境下运行，不需要任何 PyTorch 依赖。如果你已习惯 TensorFlow 生态，并且不想额外安装 PyTorch，就可以选用其中之一。

---

综上所述：

- **SB3 必须依赖 PyTorch**，([Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/?utm_source=chatgpt.com))([GitHub](https://github.com/DLR-RM/stable-baselines3?utm_source=chatgpt.com))  
- 如果只想用 TensorFlow，请切换到 **TF-Agents** 或 **Keras-RL2** 等 TensorFlow 原生库([PyPI](https://pypi.org/project/tf-agents/?utm_source=chatgpt.com))([GitHub](https://github.com/wookayin/tensorflow-agents?utm_source=chatgpt.com))。
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import ta

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_device
import matplotlib.pyplot as plt
import vectorbt as vbt


class TradingEnv(gym.Env):

    def __init__(self, csv_file, max_steps=200):
        super().__init__()

        df = pd.read_csv(csv_file)
        df = df.dropna().reset_index(drop=True)

        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd_diff()
        df.dropna(inplace=True)

        self.df = df.reset_index(drop=True)
        self.close = df['Close'].values
        self.rsi = df['rsi'].values
        self.macd = df['macd'].values
        self.max_steps = min(max_steps, len(df) - 1)

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(3, ),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.profit = []
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return np.array([
            self.close[self.current_step], self.rsi[self.current_step],
            self.macd[self.current_step]
        ],
                        dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        truncated = False

        price = self.close[self.current_step]
        self.current_step += 1

        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 1:  # Sell
            reward = price - self.entry_price
            self.position = 0
            self.profit.append(reward)
        elif action == 0 and self.position == 1:
            reward = price - self.entry_price  # floating PnL

        done = self.current_step >= self.max_steps
        obs = self._get_obs()
        return obs, reward, done, truncated, {}

    def render(self):
        print(
            f"Step {self.current_step}, Price: {self.close[self.current_step]:.2f}, Position: {self.position}"
        )


# ============ Training =====================
def train():
    env = DummyVecEnv([lambda: TradingEnv("btc.csv")])
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, device=get_device())
    model.learn(total_timesteps=20000)
    model.save("lstm_trading_model")


# ============ Evaluation with vectorbt ==============
def evaluate():
    df = pd.read_csv("btc.csv")
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd_diff()
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    env = TradingEnv("btc.csv")
    model = RecurrentPPO.load("lstm_trading_model")

    obs, _ = env.reset()
    lstm_state = None
    actions = []
    prices = []

    for _ in range(env.max_steps):
        action, lstm_state = model.predict(obs,
                                           state=lstm_state,
                                           episode_start=np.array([False]))
        obs, reward, done, truncated, _ = env.step(action)
        prices.append(env.close[env.current_step])
        actions.append(int(action))
        if done:
            break

    df = df.iloc[:len(actions)]
    df['action'] = actions
    entries = df['action'] == 1
    exits = df['action'] == 2

    pf = vbt.Portfolio.from_signals(close=df['Close'],
                                    entries=entries,
                                    exits=exits,
                                    direction="longonly",
                                    init_cash=10000)
    pf.plot().show()


if __name__ == "__main__":
    train()
    evaluate()
