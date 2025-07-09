import os
import multiprocessing
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from env import BalancingCartEnv
from typing import Dict, Any
import time
import argparse
import numpy as np

# --- 配置参数 ---
ALGORITHMS = ["PPO", "SAC", "DDPG"]
NEURON_CONFIGS = [32, 64, 128]
N_ENVS = 40
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 100_000
PARALLEL_TRAINING = False  # 默认顺序训练，可通过命令行参数改为并行
EVAL_EPISODES = 10  # 评估时的episode数量

# --- 路径设置 ---
BASE_LOG_PATH = "logspp"
BASE_MODEL_PATH = "checkpoints"
os.makedirs(BASE_LOG_PATH, exist_ok=True)
os.makedirs(BASE_MODEL_PATH, exist_ok=True)

# --- 算法超参数配置 ---
def get_hyperparams(algo: str, policy_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """返回不同算法的超参数配置"""
    common_params = {
        "policy": "MlpPolicy",
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "gamma": 0.99,
    }

    if algo == "PPO":
        return {
            **common_params,
            "n_steps": 2048,
            "batch_size": 64,
            "gae_lambda": 0.95,
            "n_epochs": 10,
            "ent_coef": 0.0,
            "learning_rate": 3e-4,
            "clip_range": 0.2,
        }
    elif algo == "SAC":
        return {
            **common_params,
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "tau": 0.005,
            "ent_coef": "auto",
        }
    elif algo == "DDPG":
        return {
            **common_params,
            "buffer_size": 200_000,
            "learning_starts": 1000,
            "batch_size": 256,
        }
    else:
        raise ValueError(f"未知算法: {algo}")

# --- 训练函数 ---
def train_model(algo: str, n_neurons: int, parallel: bool = False):
    """训练单个模型"""
    start_time = time.time()
    model_name = f"{algo}_{n_neurons}neurons"

    print(f"\n=== 开始训练 {model_name} {'(并行)' if parallel else ''} ===")

    # 创建环境（每个训练有自己的环境）
    env = make_vec_env(BalancingCartEnv, n_envs=N_ENVS)

    # 网络架构配置
    policy_kwargs = {
        "net_arch": {
            "pi": [n_neurons],
            "qf": [n_neurons],
        }
    }

    # 获取超参数并创建模型
    hyperparams = get_hyperparams(algo, policy_kwargs)
    if algo == "PPO":
        model = PPO(env=env, tensorboard_log=BASE_LOG_PATH, **hyperparams)
    elif algo == "SAC":
        model = SAC(env=env, tensorboard_log=BASE_LOG_PATH, **hyperparams)
    elif algo == "DDPG":
        model = DDPG(env=env, tensorboard_log=BASE_LOG_PATH, **hyperparams)

    # 设置保存路径和回调
    model_save_path = os.path.join(BASE_MODEL_PATH, model_name)
    os.makedirs(model_save_path, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=model_save_path,
        name_prefix=model_name
    )

    # 训练模型
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=model_name
    )

    # 保存最终模型
    final_model_path = os.path.join(model_save_path, "final_model")
    model.save(final_model_path)

    env.close()

    duration = time.time() - start_time
    print(f"=== 训练完成 {model_name} | 耗时: {duration:.2f}s | 模型保存到 {final_model_path} ===")

# --- 评估函数 ---
def evaluate_model(algo: str, n_neurons: int):
    """评估单个模型"""
    model_name = f"{algo}_{n_neurons}neurons"
    model_path = os.path.join(BASE_MODEL_PATH, model_name, "final_model")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"模型 {model_name} 不存在，跳过评估")
        return None
    
    print(f"\n=== 开始评估 {model_name} ===")
    
    # 加载模型
    if algo == "PPO":
        model = PPO.load(model_path)
    elif algo == "SAC":
        model = SAC.load(model_path)
    elif algo == "DDPG":
        model = DDPG.load(model_path)
    
    # 创建评估环境
    eval_env = make_vec_env(BalancingCartEnv, n_envs=N_ENVS)
    
    # 评估模型
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True
    )
    
    eval_env.close()
    
    print(f"=== 评估完成 {model_name} ===")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print("="*50)
    
    return mean_reward, std_reward

# --- 评估所有模型 ---
def evaluate_all_models():
    """评估所有训练好的模型"""
    results = {}
    
    print("\n" + "="*50)
    print("开始评估所有模型")
    print("="*50 + "\n")
    
    for algo in ALGORITHMS:
        for n_neurons in NEURON_CONFIGS:
            result = evaluate_model(algo, n_neurons)
            if result is not None:
                results[f"{algo}_{n_neurons}neurons"] = result
    
    # 打印评估结果汇总
    print("\n" + "="*50)
    print("评估结果汇总:")
    print("="*50)
    for model_name, (mean_reward, std_reward) in results.items():
        print(f"{model_name}: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 找出最佳模型
    if results:
        best_model = max(results.items(), key=lambda x: x[1][0])
        print("\n" + "="*50)
        print(f"最佳模型: {best_model[0]}")
        print(f"平均奖励: {best_model[1][0]:.2f} ± {best_model[1][1]:.2f}")
        print("="*50)
    
    return results

# --- 并行训练函数 ---
def parallel_train():
    """并行训练所有组合"""
    processes = []

    for algo in ALGORITHMS:
        for n_neurons in NEURON_CONFIGS:
            p = multiprocessing.Process(
                target=train_model,
                args=(algo, n_neurons, True)
            )
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

# --- 顺序训练函数 ---
def sequential_train():
    """顺序训练所有组合"""
    for algo in ALGORITHMS:
        for n_neurons in NEURON_CONFIGS:
            train_model(algo, n_neurons)

# --- 主函数 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="强化学习训练脚本")
    parser.add_argument("--no_parallel", action="store_true", help="使用并行训练模式")
    parser.add_argument("--eval", action="store_true", help="仅评估模型而不训练")
    args = parser.parse_args()

    if args.eval:
        evaluate_all_models()
    else:
        print("\n" + "="*50)
        print(f"开始训练所有组合 (算法: {ALGORITHMS}, 神经元配置: {NEURON_CONFIGS})")
        print(f"训练模式: {'顺序' if args.no_parallel else '并行'}")
        print("="*50 + "\n")

        if args.no_parallel:
            sequential_train()
        else:
            parallel_train()

        print("\n" + "="*50)
        print("所有训练任务完成!")
        print("="*50)
        
        # 训练完成后自动评估
        evaluate_all_models()