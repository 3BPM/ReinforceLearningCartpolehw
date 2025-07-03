import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import cont2discrete
import pygame

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
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Balancing Cart")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        else:
            self.screen = None

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
        if self.render_mode != "human" or self.state is None:
            return

        self.screen.fill((255, 255, 255))
        theta_L, theta_R, theta_1, theta_2, _, _, _, _ = self.state
        
        # World to screen scaling
        world_width = 4.0
        scale = self.screen.get_width() / world_width
        cart_y_pos = self.screen.get_height() * 0.7
        cart_x_world = self.r * (theta_L + theta_R) / 2.0
        cart_x_screen = int(self.screen.get_width() / 2 + cart_x_world * scale)
        
        # Draw cart body
        cart_width_pixels = int(0.2 * scale)
        cart_height_pixels = int(0.1 * scale)
        l, r_coord, t, b = -cart_width_pixels/2, cart_width_pixels/2, cart_height_pixels/2, -cart_height_pixels/2
        cart_coords = [(l, b), (l, t), (r_coord, t), (r_coord, b)]
        rotated_cart_coords = []
        for x_rel, y_rel in cart_coords:
            x_rot = x_rel * np.cos(theta_1) - y_rel * np.sin(theta_1)
            y_rot = x_rel * np.sin(theta_1) + y_rel * np.cos(theta_1)
            rotated_cart_coords.append((int(cart_x_screen + x_rot), int(cart_y_pos - y_rot)))
        pygame.draw.polygon(self.screen, (0, 0, 0), rotated_cart_coords)
        
        # Draw wheels
        # 修改wheel位置计算（约第290行）
        wheel_radius_pixels = int(self.r * scale)
        wheel_offset_x = cart_width_pixels / 2 - wheel_radius_pixels

        # 修正Y坐标计算（关键修改！）
        w_rel_y = - (cart_height_pixels / 2 + wheel_radius_pixels)  # 负号表示下方

        lw_rel_x = -wheel_offset_x
        rw_rel_x = wheel_offset_x

        # 保持旋转计算不变
        lw_rot_x = lw_rel_x * np.cos(theta_1) - w_rel_y * np.sin(theta_1)
        lw_rot_y = lw_rel_x * np.sin(theta_1) + w_rel_y * np.cos(theta_1)
        rw_rot_x = rw_rel_x * np.cos(theta_1) - w_rel_y * np.sin(theta_1)
        rw_rot_y = rw_rel_x * np.sin(theta_1) + w_rel_y * np.cos(theta_1)

        # 更新绘制位置
        left_wheel_pos = (int(cart_x_screen + lw_rot_x), int(cart_y_pos - lw_rot_y))
        right_wheel_pos = (int(cart_x_screen + rw_rot_x), int(cart_y_pos - rw_rot_y))
        # wheel_radius_pixels = int(self.r * scale)
        # wheel_offset_x = cart_width_pixels / 2 - wheel_radius_pixels
        
        # lw_rel_x = -wheel_offset_x
        # rw_rel_x = wheel_offset_x
        # w_rel_y = cart_height_pixels / 2 + wheel_radius_pixels
        
        # lw_rot_x = lw_rel_x * np.cos(theta_1) - w_rel_y * np.sin(theta_1)
        # lw_rot_y = lw_rel_x * np.sin(theta_1) + w_rel_y * np.cos(theta_1)
        # rw_rot_x = rw_rel_x * np.cos(theta_1) - w_rel_y * np.sin(theta_1)
        # rw_rot_y = rw_rel_x * np.sin(theta_1) + w_rel_y * np.cos(theta_1)

        # left_wheel_pos = (int(cart_x_screen + lw_rot_x), int(cart_y_pos - lw_rot_y))
        # right_wheel_pos = (int(cart_x_screen + rw_rot_x), int(cart_y_pos - rw_rot_y))
        pygame.draw.circle(self.screen, (100, 100, 100), left_wheel_pos, wheel_radius_pixels)
        pygame.draw.circle(self.screen, (100, 100, 100), right_wheel_pos, wheel_radius_pixels)
        
        # Draw pendulum
        pendulum_length_pixels = int(self.L_2 * scale * 0.8)
        pendulum_end_x = int(cart_x_screen + pendulum_length_pixels * np.sin(theta_2))
        pendulum_end_y = int(cart_y_pos - pendulum_length_pixels * np.cos(theta_2))
        pygame.draw.line(self.screen, (200, 0, 0), (cart_x_screen, cart_y_pos), 
                         (pendulum_end_x, pendulum_end_y), 5)
        
        # Display info
        text_lines = [
            f"Step: {self.current_step}/{self.max_episode_steps}",
            f"Body (th1): {theta_1 * 180/np.pi:.1f} deg",
            f"Pend (th2): {theta_2 * 180/np.pi:.1f} deg",
            f"Wheel L: {theta_L:.1f} rad", 
            f"Wheel R: {theta_R:.1f} rad"
        ]
        for i, line in enumerate(text_lines):
            surf = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(surf, (10, 10 + i*20))

        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.render_mode == "human" and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None