
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(DQN, self).__init__()
#         # 选项1: 当前结构 (128-128)
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, action_dim)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # 选项1: 原始结构 (128-128) - 已被替换
        # self.fc1 = nn.Linear(state_dim, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, action_dim)

        # 新结构: 32x32
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)
        # ...

        # 选项2: 更小/更简单的网络 (有时对于这类问题效果不错或更容易训练)
        # self.fc1 = nn.Linear(state_dim, 64)
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, action_dim)

        # 选项3: 稍微大一点的网络 (如果问题确实复杂)
        # self.fc1 = nn.Linear(state_dim, 256)
        # self.fc2 = nn.Linear(256, 128) # 可以尝试非对称
        # self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, env,
                 gamma=0.99,            # 折扣因子
                 lr=1e-4,               # 学习率 (可以尝试 5e-4, 2e-4, 1e-4)
                 batch_size=64,         # 批量大小 (可以尝试 32, 128)
                 memory_size=100000,    # Replay Buffer 大小
                 epsilon_start=1.0,     # 初始 Epsilon
                 epsilon_end=0.01,      # 最终 Epsilon (可以尝试 0.05 或 0.1 如果过早收敛到次优)
                 epsilon_decay=0.995,   # Epsilon 衰减率 (可以尝试 0.99 或 0.999 根据训练时长调整)
                 target_update_freq=10): # 目标网络更新频率 (由训练脚本控制)
        self.env = env
        self.state_dim = env.observation_space.shape[0]

        # 动作离散化 (保持与之前相同的离散化)
        # 你可以尝试增加或减少离散化的级别数，但这会改变 action_dim
        # 例如，7个级别: np.linspace(-env.max_wheel_angular_accel, env.max_wheel_angular_accel, 7)
        # 这将导致 action_dim = 7 * 7 = 49
        self.accel_levels = np.linspace(-env.max_wheel_angular_accel,
                                       env.max_wheel_angular_accel, 5) # 当前是5个级别
        self.num_action_levels_per_wheel = len(self.accel_levels)
        self.action_dim = self.num_action_levels_per_wheel ** 2 # 每个轮子独立离散化，总动作数是级别数的平方

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 网络
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network设置为评估模式

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # 可以尝试 RMSprop: self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)

        # 训练参数
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def get_continuous_action(self, discrete_action_idx):
        """ 将离散动作索引转换为连续的双轮加速度 """
        if not (0 <= discrete_action_idx < self.action_dim):
            raise ValueError(f"discrete_action_idx {discrete_action_idx} out of range [0, {self.action_dim-1}]")

        idx_uL = discrete_action_idx // self.num_action_levels_per_wheel
        idx_uR = discrete_action_idx % self.num_action_levels_per_wheel
        return np.array([self.accel_levels[idx_uL], self.accel_levels[idx_uR]], dtype=np.float32)

    def remember(self, state, action_idx, reward, next_state, done):
        """ 存储经验到Replay Buffer """
        self.memory.append((state, action_idx, reward, next_state, done))

    def act(self, state):
        """ 根据当前状态和epsilon-greedy策略选择一个离散动作索引 """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1) # 随机探索

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): # 在推断时不计算梯度
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item() # 选择Q值最大的动作

    def update_epsilon(self):
        """ 更新Epsilon值 """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """ 将策略网络的权重复制到目标网络 """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def replay(self):
        """ 从Replay Buffer中采样并训练策略网络 """
        if len(self.memory) < self.batch_size:
            return 0.0 # Buffer中的样本不足以进行训练

        # 从memory中随机采样一个batch
        batch = random.sample(self.memory, self.batch_size)
        # 解包batch中的数据
        states, action_indices, rewards, next_states, dones = zip(*batch)

        # 将Python list/tuple 转换为Numpy数组，然后再转换为Torch张量
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        action_indices_tensor = torch.LongTensor(np.array(action_indices)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.FloatTensor(np.array(dones)).to(self.device) # Bool 转 Float (0.0 or 1.0)

        # 计算当前状态下所采取动作的Q值: Q(s, a)
        # policy_net(states_tensor) 输出所有动作的Q值，shape: (batch_size, action_dim)
        # gather(1, action_indices_tensor.unsqueeze(1)) 从第二个维度（dim=1）按action_indices_tensor选择Q值
        current_q_values = self.policy_net(states_tensor).gather(1, action_indices_tensor.unsqueeze(1))

        # 计算下一个状态的最大Q值: max_a' Q_target(s', a')
        # target_net(next_states_tensor) 输出所有动作的Q值
        # .max(1) 返回 (最大值, 最大值对应的索引)，我们只需要最大值 [0]
        # .detach() 从计算图中分离，目标Q值不参与梯度回传
        next_q_values_target_net = self.target_net(next_states_tensor).max(1)[0].detach()

        # 计算目标Q值: r + gamma * max_a' Q_target(s', a') if not done else r
        # (1 - dones_tensor) 处理终止状态，如果done=1，则后续Q值为0
        target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values_target_net

        # 计算损失 (MSE Loss)
        # current_q_values.squeeze(1) 将shape从 (batch_size, 1) 变为 (batch_size) 以匹配 target_q_values
        loss = nn.MSELoss()(current_q_values.squeeze(1), target_q_values)

        # 反向传播和优化
        self.optimizer.zero_grad() # 清除旧梯度
        loss.backward()           # 计算新梯度
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) # 可选: 梯度裁剪防止梯度爆炸
        self.optimizer.step()      # 更新网络权重

        return loss.item() # 返回loss值用于监控

    def save(self, filepath="dqn_model.pth"):
        """ 保存模型状态 """
        # 确保目录存在 (如果filepath包含目录)
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(), # 保存目标网络通常不是必须的，但可以保存
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            # 可以添加其他需要保存的参数，如训练回合数等
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath="dqn_model.pth"):
        """ 加载模型状态 """
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device) # map_location确保模型加载到正确的设备
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            if 'target_net_state_dict' in checkpoint: # 向后兼容，如果旧模型没有保存target_net
                 self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            else: # 如果没有保存，则从policy_net复制
                self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon',    0.1) #如果模型中没存epsilon，用默认值
            self.policy_net.train() # 确保policy_net处于训练模式
            self.target_net.eval()  # 确保target_net处于评估模式
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}, starting from scratch.")