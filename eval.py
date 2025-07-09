# enjoy.py

import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from env import BalancingCartEnv
import time

# --- 配置 ---
ALGO_TO_USE = "SAC" # 必须与训练时使用的算法一致
MODEL_PATH = f"models/{ALGO_TO_USE}_balancing_cart/final_model.zip" # 要加载的模型路径

# --- 加载算法 ---
algo_map = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG}
SELECTED_ALGO = algo_map[ALGO_TO_USE]

# --- 创建环境并加载模型 ---
env = BalancingCartEnv(render_mode="human")
model = SELECTED_ALGO.load(MODEL_PATH, env=env)

# --- 运行评估 ---
vec_env = model.get_env()
obs = vec_env.reset()
episodes = 5
for i in range(episodes):
    done = False
    episode_reward = 0
    print(f"\n--- Starting Episode {i+1} ---")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        result = vec_env.step(action)
        obs, reward, done, info = result
        episode_reward += reward[0]
        vec_env.render()  # 渲染环境
        if done:
            print(f"Episode finished. Reward: {episode_reward:.2f}")
            obs = vec_env.reset() # 自动重置

env.close()
