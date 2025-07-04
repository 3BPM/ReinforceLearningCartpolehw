import numpy as np
import time
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_decay=0.9999, epsilon_min=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Enhanced state discretization
        self.theta1_bins = np.linspace(-0.5, 0.5, 11)     # ±28°, 11 bins
        self.theta2_bins = np.linspace(-0.5, 0.5, 21)     # ±28°, 21 bins
        self.dot_theta1_bins = np.linspace(-2.0, 2.0, 11)  # 11 bins
        self.dot_theta2_bins = np.linspace(-2.0, 2.0, 21) # 21 bins

        self.num_bins = [
            len(self.theta1_bins) + 1,  # +1 for out-of-range values
            len(self.theta2_bins) + 1,
            len(self.dot_theta1_bins) + 1,
            len(self.dot_theta2_bins) + 1
        ]

        # Enhanced action discretization (5 levels per wheel)
        self.accel_levels = np.linspace(-env.max_wheel_angular_accel, 
                                       env.max_wheel_angular_accel, 5)
        self.num_action_levels_per_wheel = len(self.accel_levels)
        self.num_discrete_actions = self.num_action_levels_per_wheel ** 2  # 25 actions

        # Initialize Q-table with improved random initialization
        self.q_table = np.random.uniform(low=-0.5, high=0, size=(*self.num_bins, self.num_discrete_actions))
        
        # Tracking for debugging
        self.state_visit_counts = defaultdict(int)
        self.episode_stats = {
            'rewards': [],
            'lengths': [],
            'td_errors': []
        }

    def discretize_state(self, continuous_state):
        # Clip and discretize each state variable
        theta_1 = np.clip(continuous_state[2], -0.5, 0.5)
        theta_2 = np.clip(continuous_state[3], -0.5, 0.5)
        dot_theta_1 = np.clip(continuous_state[6], -2.0, 2.0)
        dot_theta_2 = np.clip(continuous_state[7], -2.0, 2.0)
        
        s1 = np.digitize(theta_1, self.theta1_bins)
        s2 = np.digitize(theta_2, self.theta2_bins)
        s3 = np.digitize(dot_theta_1, self.dot_theta1_bins)
        s4 = np.digitize(dot_theta_2, self.dot_theta2_bins)
        
        # Track state visits for debugging
        self.state_visit_counts[(s1, s2, s3, s4)] += 1
        
        return (s1, s2, s3, s4)

    def get_continuous_action(self, discrete_action_idx):
        idx_uL = discrete_action_idx // self.num_action_levels_per_wheel
        idx_uR = discrete_action_idx % self.num_action_levels_per_wheel
        return np.array([self.accel_levels[idx_uL], self.accel_levels[idx_uR]], dtype=np.float32)

    def choose_action(self, discrete_state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_discrete_actions)  # Explore
        return np.argmax(self.q_table[discrete_state])  # Exploit

    def update_q_table(self, s, a, r, s_prime, terminated):
        s_discrete = self.discretize_state(s)
        s_prime_discrete = self.discretize_state(s_prime)

        current_q = self.q_table[s_discrete + (a,)]
        if terminated:
            target_q = r
        else:
            target_q = r + self.gamma * np.max(self.q_table[s_prime_discrete])

        # Track TD error for debugging
        td_error = target_q - current_q
        self.episode_stats['td_errors'].append(abs(td_error))
        
        # Update Q-value
        self.q_table[s_discrete + (a,)] = current_q + self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, num_episodes, render_every_n=500, save_every_n=5000):
        start_time = time.time()
        
        for episode in range(num_episodes):
            continuous_s, _ = self.env.reset()
            terminated = truncated = False
            episode_reward = 0
            
            while not (terminated or truncated):
                s_discrete = self.discretize_state(continuous_s)
                discrete_a = self.choose_action(s_discrete)
                continuous_a = self.get_continuous_action(discrete_a)
                
                continuous_s_prime, reward, terminated, truncated, _ = self.env.step(continuous_a)
                
                self.update_q_table(continuous_s, discrete_a, reward, continuous_s_prime, terminated)
                continuous_s = continuous_s_prime
                episode_reward += reward
                
                if self.env.render_mode == "human" and episode % render_every_n == 0:
                    self.env.render()

            # Post-episode updates
            self.decay_epsilon()
            self.episode_stats['rewards'].append(episode_reward)
            self.episode_stats['lengths'].append(self.env.current_step)
            
            # Periodic logging
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_stats['rewards'][-100:])
                avg_length = np.mean(self.episode_stats['lengths'][-100:])
                avg_td_error = np.mean(self.episode_stats['td_errors'][-1000:]) if self.episode_stats['td_errors'] else 0
                elapsed_time = time.time() - start_time
                
                print(f"Ep {episode+1}/{num_episodes} | "
                      f"Avg R: {avg_reward:.1f} | "
                      f"Avg L: {avg_length:.0f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Avg TD: {avg_td_error:.3f} | "
                      f"Time: {elapsed_time:.1f}s")
                
                self.episode_stats['td_errors'] = []  # Reset TD error tracking
            
            # Save Q-table periodically
            if (episode + 1) % save_every_n == 0:
                np.save(f'q_table_ep{episode+1}.npy', self.q_table)
        
        return self.episode_stats['rewards'], self.episode_stats['lengths']