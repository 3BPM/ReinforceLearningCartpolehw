# train.py

import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from env import BalancingCartEnv

# --- 配置 ---
ALGO_TO_USE = "PPO"  # 在这里切换你想用的算法: "PPO", "SAC", "DDPG"
TOTAL_TIMESTEPS = 300_000 # 总训练步数
MODEL_NAME = f"{ALGO_TO_USE}_balancing_cart"

# --- 路径设置 ---
log_path = os.path.join("logs", MODEL_NAME)
model_save_path = os.path.join("models", MODEL_NAME)
os.makedirs(log_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)

# --- 训练环境 ---
# 使用 make_vec_env 来创建（可能是并行的）环境
# 对于复杂任务，使用多个并行环境可以加速训练
env = make_vec_env(BalancingCartEnv, n_envs=4)

# --- 算法和超参数选择 ---
# 为不同算法设置一些合理的默认超参数
hyperparams = {
    "PPO": {"policy": "MlpPolicy", "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95},
    "SAC": {"policy": "MlpPolicy", "buffer_size": 200_000, "learning_starts": 1000, "batch_size": 256, "gamma": 0.99},
    "DDPG": {"policy": "MlpPolicy", "buffer_size": 200_000, "learning_starts": 1000, "batch_size": 256, "gamma": 0.99}
}

algo_map = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG}
SELECTED_ALGO = algo_map[ALGO_TO_USE]
SELECTED_HYPERPARAMS = hyperparams[ALGO_TO_USE]

# --- 回调函数：用于在训练过程中自动保存模型 ---
checkpoint_callback = CheckpointCallback(
    save_freq=10_000, # 每10000步保存一次
    save_path=model_save_path,
    name_prefix="rl_model"
)

# --- 模型初始化和训练 ---
model = SELECTED_ALGO(
    env=env,
    verbose=1,
    tensorboard_log=log_path,
    **SELECTED_HYPERPARAMS
)

print(f"--- Starting training for {ALGO_TO_USE} ---")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback
)

# --- 保存最终模型 ---
final_model_path = os.path.join(model_save_path, "final_model")
model.save(final_model_path)
print(f"--- Training finished. Final model saved to {final_model_path} ---")

env.close()