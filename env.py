
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import cont2discrete
import pygame
from envsim.config import Config
from envsim.lqr_controller import build_system_matrices
from envsim.renderer import UnicycleRenderer

class BalancingCartEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # 系统矩阵
        A_c, B_c, _, _, self.A_d, self.B_d  = build_system_matrices(Ts=Config.dt) 
        self.Ts = 0.01  # 控制周期，与单片机代码一致
        self.max_wheel_angular_accel = Config.max_wheel_angular_accel

        # 代码是离散的，这里我们假设A_c, B_c是连续时间矩阵
        # 离散化


        # Action space 连续的
        self.action_space = spaces.Box(
            low=-Config.max_wheel_angular_accel,
            high=Config.max_wheel_angular_accel,
            shape=(2,),
            dtype=np.float32
        )

        # Observation space
        # [theta_L, theta_R, theta_1, theta_2, dot_theta_L, dot_theta_R, dot_theta_1, dot_theta_2]
        high_obs = np.array([
            np.inf, np.inf,                         # Wheel positions (rad)
            np.pi / 2, np.pi / 2,                   # Body and pole angles (rad)
            100.0, 100.0,                           # Wheel velocities (rad/s)
            100.0, 100.0                            # Body and pole angular velocities (rad/s)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

        # Episode control
        self.theta1_limit = 25 * np.pi / 180 # 25度
        self.theta2_limit = 25 * np.pi / 180 # 25度
        self.max_episode_steps = 10/self.Ts # 相当于10秒
        self.current_step = 0
        self.state = None

        # Rendering
        self.render_mode = render_mode
        self.renderer = None
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 800))
            self.clock = pygame.time.Clock()
            self.renderer = UnicycleRenderer()


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 动力学更新
        self.state = self.A_d @ self.state + self.B_d @ action
        theta_L, theta_R, theta_1, theta_2, dot_theta_L, dot_theta_R, dot_theta_1, dot_theta_2 = self.state

        # 终止条件
        terminated = bool(
            abs(theta_1) > self.theta1_limit or
            abs(theta_2) > self.theta2_limit
        )

        # --- 奖励函数 (核心！) ---
        # 目标是让 theta_1 和 theta_2 都趋近于 0
        if not terminated:
            # 存活奖励
            reward = 1.0
            
            # 角度惩罚: 离垂直线越远，惩罚越大
            angle_penalty = 5.0 * (theta_1**2) + 10.0 * (theta_2**2)
            
            # 速度惩罚: 不希望它晃动得太厉害
            velocity_penalty = 0.01 * (dot_theta_1**2) + 0.1 * (dot_theta_2**2)

            # 动作惩罚: 节省能量
            action_penalty = 0.001 * np.sum(np.square(action))
            
            reward -= (angle_penalty + velocity_penalty + action_penalty)
        else:
            # 失败惩罚
            reward = -100.0

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return self.state.astype(np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, state=None):
        super().reset(seed=seed)
        # 从一个稍微偏离平衡点的位置开始
        if state is None:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=8)
        else:
            self.state=state
        self.current_step = 0
        
        # 确保初始角度不为0，增加挑战
        self.state[2] = self.np_random.uniform(low=-0.1, high=0.1) # theta_1
        self.state[3] = self.np_random.uniform(low=-0.1, high=0.1) # theta_2
        
        if self.render_mode == "human":
            self.render()
            
        return self.state.astype(np.float32), {}

    def render(self):
        if self.render_mode != "human" or self.renderer is None:
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        theta_L, theta_R, theta_1, theta_2, _, _, _, _ = self.state
        self.renderer.render_cartpole(self.screen, self.state, theta_L, theta_R, theta_1, theta_2)
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None