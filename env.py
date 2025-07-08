import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import cont2discrete
import pygame
from envsim.config import Config
from envsim.lqr_controller import build_system_matrices
from envsim.renderer import UnicycleRenderer  # 导入渲染器

class BalancingCartEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # 统一物理参数来源
        self.m_1 = Config.m_car
        self.m_2 = Config.m_pole
        self.r = Config.r_wheel
        self.L_1 = Config.L_body
        self.L_2 = Config.L_pole
        self.l_1 = Config.l_body
        self.l_2 = Config.l_pole
        self.g = Config.g
        self.I_1_com = Config.I_body
        self.I_2_com = Config.I_pole

        # 系统矩阵（与LQRController一致）
        A_c, B_c, _, _, _, _ = build_system_matrices(Ts=0.005)
        self.A_c = A_c
        self.B_c = B_c
        self.Ts = 0.005  # Sampling time
        # 离散化
        discrete_system = cont2discrete((self.A_c, self.B_c, np.eye(8), np.zeros((8, 2))),
                                       self.Ts, method='zoh')
        self.A_d = discrete_system[0]
        self.B_d = discrete_system[1]

        # Action space
        self.max_wheel_angular_accel = 50.0
        self.action_space = spaces.Box(
            low=np.full((2,), -self.max_wheel_angular_accel, dtype=np.float32),
            high=np.full((2,), self.max_wheel_angular_accel, dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Observation space
        obs_limit_angles = 90 * np.pi / 180
        obs_limit_vels = 100
        obs_limit_wheel_pos = 1000
        high_obs = np.array([
            obs_limit_wheel_pos, obs_limit_wheel_pos,
            obs_limit_angles, obs_limit_angles,
            obs_limit_vels, obs_limit_vels, obs_limit_vels, obs_limit_vels
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

        # Episode control
        self.theta1_limit = 60 * np.pi / 180
        self.theta2_limit = 60 * np.pi / 180
        self.max_episode_steps = 4000
        self.current_step = 0
        self.state = None

        # Rendering
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((1200,800))
            self.renderer = UnicycleRenderer()  # 初始化渲染器
        else:
            self.renderer = None

    def step(self, action):
        # Clip action
        if isinstance(self.action_space, spaces.Box):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        # Dynamics update
        if self.state is not None and action is not None:
            self.state = self.A_d @ self.state + self.B_d @ action
            theta_L, theta_R, theta_1, theta_2, dot_theta_L, dot_theta_R, dot_theta_1, dot_theta_2 = self.state
        else:
            # fallback to zeros if state/action is None
            theta_L = theta_R = theta_1 = theta_2 = dot_theta_L = dot_theta_R = dot_theta_1 = dot_theta_2 = 0.0

        # Termination condition
        terminated = bool(
            abs(theta_1) > self.theta1_limit or
            abs(theta_2) > self.theta2_limit
        )

        # Reward function (改进版)
        reward = 1.0  # Survival bonus
        if not terminated:
            angle_penalty = 0.5*(theta_1**2) + 5*(theta_2**2)
            velocity_penalty = 0.001 * (dot_theta_1**2 + dot_theta_2**2)
            action_penalty = 0.0001 * (action[0]**2 + action[1]**2)
            upright_reward = 0.5 * np.exp(-20 * theta_2**2)

            reward += upright_reward - (angle_penalty + velocity_penalty + action_penalty)
        else:
            reward = -100.0  # Failure penalty

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        if self.state is not None:
            return self.state.astype(np.float32), reward, terminated, truncated, {}
        else:
            return np.zeros(8, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, state=None):
        super().reset(seed=seed)
        initial_theta1 = self.np_random.uniform(low=-0.1, high=0.1)
        initial_theta2 = self.np_random.uniform(low=-0.1, high=0.1)
        if state is not None:
            self.state = state
        else:
            self.state = np.array([
                0.0, 0.0,
                initial_theta1, initial_theta2,
                0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)

        self.current_step = 0
        if self.render_mode == "human":
            self.render()
        return self.state.astype(np.float32), {}

    def render(self):
        if self.render_mode != "human" or self.state is None or self.renderer is None:
            return

        # 使用渲染器进行渲染
        theta_L, theta_R, theta_1, theta_2, _, _, _, _ = self.state
        self.renderer.render_cartpole(self.screen, self.state, theta_L, theta_R, theta_1, theta_2)

        # Display info
        text_lines = [
            f"Step: {self.current_step}/{self.max_episode_steps}",
            f"Body (th1): {theta_1 * 180/np.pi:.1f} deg",
            f"Pend (th2): {theta_2 * 180/np.pi:.1f} deg",
            f"Wheel L: {theta_L:.1f} rad",
            f"Wheel R: {theta_R:.1f} rad"
        ]
        # for i, line in enumerate(text_lines):
        #     surf = self.font.render(line, True, (0, 0, 0))
        #     self.screen.blit(surf, (10, 10 + i*20))
        print("\n".join(text_lines))  # For debugging, you can print to console
        pygame.display.flip()
        # self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.render_mode == "human" and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None