import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, env, gamma=0.99, lr=1e-4, batch_size=64,
                 memory_size=100000, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, hidden_size=256):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = 25  # Keep the same discrete action space
        self.hidden_size = hidden_size

        # 网络
        self.policy_net = DQN(self.state_dim, self.action_dim, hidden_size)
        self.target_net = DQN(self.state_dim, self.action_dim, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 训练参数
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        # 动作离散化 (保持与之前QLearning相同的离散化)
        self.accel_levels = np.linspace(-env.max_wheel_angular_accel,
                                       env.max_wheel_angular_accel, 5)
        self.num_action_levels_per_wheel = len(self.accel_levels)

    def get_continuous_action(self, discrete_action_idx):
        idx_uL = discrete_action_idx // self.num_action_levels_per_wheel
        idx_uR = discrete_action_idx % self.num_action_levels_per_wheel
        return np.array([self.accel_levels[idx_uL], self.accel_levels[idx_uR]], dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def quantize_to_int8(self):
        """
        Applies post-training dynamic quantization to the policy network.
        """
        if self.device.type == 'cuda':
            print("Quantization is not supported on CUDA, moving model to CPU.")
            self.policy_net.to('cpu')

        self.policy_net.eval()  # Set the model to evaluation mode

        # Apply dynamic quantization to the linear layers of the policy network
        self.policy_net = torch.quantization.quantize_dynamic(
            self.policy_net, {nn.Linear}, dtype=torch.qint8
        )
        torch.save(self.policy_net.state_dict(), 'quantized_policy_net.pth')
        print("Model has been quantized to INT8.")


    def save(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']