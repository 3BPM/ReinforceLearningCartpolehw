import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import cont2discrete
import pygame
from envsim.renderer import UnicycleRenderer  # 导入渲染器

class BalancingCartEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # Physical parameters
        self.m_1 = 0.9      # kg
        self.m_2 = 0.1      # kg
        self.r = 0.0335     # m
        self.L_1 = 0.126    # m
        self.L_2 = 0.390    # m
        self.l_1 = self.L_1 / 2
        self.l_2 = self.L_2 / 2
        self.g = 9.81       # m/s^2

        # Moment of inertia
        self.I_1_com = (1/12) * self.m_1 * self.L_1**2
        self.I_2_com = (1/12) * self.m_2 * self.L_2**2

        # System matrices
        p = np.zeros((4, 4))
        p[0, 0] = 1.0
        p[1, 1] = 1.0
        p[2, 0] = (self.r / 2) * (self.m_1 * self.l_1 + self.m_2 * self.L_1)
        p[2, 1] = (self.r / 2) * (self.m_1 * self.l_1 + self.m_2 * self.L_1)
        p[2, 2] = self.m_1 * self.l_1**2 + self.m_2 * self.L_1**2 + self.I_1_com
        p[2, 3] = self.m_2 * self.L_1 * self.l_2
        p[3, 0] = (self.r / 2) * self.m_2 * self.l_2
        p[3, 1] = (self.r / 2) * self.m_2 * self.l_2
        p[3, 2] = self.m_2 * self.L_1 * self.l_2
        p[3, 3] = self.m_2 * self.l_2**2 + self.I_2_com
        self.P_matrix = p
        self.P_inv_matrix = np.linalg.inv(self.P_matrix)

        q_coeffs = np.zeros((4, 10))
        q_coeffs[0, 8] = 1.0
        q_coeffs[1, 9] = 1.0
        q_coeffs[2, 2] = (self.m_1 * self.l_1 + self.m_2 * self.L_1) * self.g
        q_coeffs[3, 3] = self.m_2 * self.g * self.l_2
        self.Q_coeffs_matrix = q_coeffs
        self.temp_matrix = self.P_inv_matrix @ self.Q_coeffs_matrix

        # Continuous state space
        self.A_c = np.zeros((8, 8))
        self.A_c[0:4, 4:8] = np.eye(4)
        self.A_c[4:8, 0:8] = self.temp_matrix[:, 0:8]
        self.B_c = np.zeros((8, 2))
        self.B_c[4:8, 0:2] = self.temp_matrix[:, 8:10]

        # Discrete time system
        self.Ts = 0.005  # Sampling time
        discrete_system = cont2discrete((self.A_c, self.B_c, np.eye(8), np.zeros((8, 2))),
                                       self.Ts, method='zoh')
        self.A_d = discrete_system[0]
        self.B_d = discrete_system[1]

        # Action space
        self.max_wheel_angular_accel = 50.0
        self.action_space = spaces.Box(
            low=-self.max_wheel_angular_accel,
            high=self.max_wheel_angular_accel,
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
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Dynamics update
        self.state = self.A_d @ self.state + self.B_d @ action
        theta_L, theta_R, theta_1, theta_2, dot_theta_L, dot_theta_R, dot_theta_1, dot_theta_2 = self.state

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

        return self.state.astype(np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_theta1 = self.np_random.uniform(low=-0.1, high=0.1)
        initial_theta2 = self.np_random.uniform(low=-0.1, high=0.1)

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