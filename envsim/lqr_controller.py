import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from config import Config

# ==================== LQR控制器类 ====================
class LQRController:
    def __init__(self):
        # 系统参数
        m_1 = Config.m_car      # 车体质量
        m_2 = Config.m_pole     # 摆杆质量
        L_1 = Config.L_pole     # 摆杆长度
        l_1 = Config.l_pole     # 摆杆质心到关节距离
        l_2 = Config.l_pole     # 摆杆质心到关节距离
        I_1 = Config.I_pole     # 摆杆惯量
        I_2 = Config.I_pole     # 摆杆惯量
        r = Config.r_wheel      # 轮子半径
        g = Config.g            # 重力加速度
        
        # 系统建模 - 构建惯性矩阵 P
        P = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [(r/2)*(m_1*l_1 + m_2*L_1), (r/2)*(m_1*l_1 + m_2*L_1), m_1*l_1**2 + m_2*L_1**2 + I_1, m_2*L_1*l_2],
            [(r/2)*m_2*l_2,              (r/2)*m_2*l_2,            m_2*L_1*l_2,               m_2*l_2**2 + I_2]
        ])


        # 系统建模 - 构建重力矩阵Q
        q = np.zeros((4, 10))
        q[2, 2] = (m_1*l_1 + m_2*L_1)*g  # 重力相关项
        q[3, 3] = m_2*g*l_2              # 重力相关项
        q[0, 8] = 1                      # 位置状态
        q[1, 9] = 1                      # 位置状态

        # 构建状态空间模型
        temp = inv(P) @ q

        # 状态矩阵A
        A = np.block([
            [np.zeros((4, 4)), np.eye(4)],  # 位置和速度的状态关系
            [temp[:, :8]]                   # 动力学关系
        ])

        # 输入矩阵B
        B = np.block([
            [np.zeros((4, 2))],             # 位置不受直接控制
            [temp[:, 8:10]]                 # 控制输入影响加速度
        ])

        # 输出矩阵C - 测量所有位置状态
        C = np.block([np.eye(4), np.zeros((4, 4))])

        # 直接传输矩阵D
        D = np.zeros((4, 2))

        # 离散化系统
        Ts = 0.01  # 采样时间(s)
        G, H, _, _, _ = cont2discrete((A, B, C, D), Ts, method='zoh')

        # 系统可控性分析
        Tc = self._controllability_matrix(G, H)
        if np.linalg.matrix_rank(Tc) == A.shape[0]:
            print(f'系统可控性分析: 系统是完全可控的(秩={np.linalg.matrix_rank(Tc)})')
        else:
            raise ValueError('系统不可控，无法设计控制器')

        # LQR控制器设计
        # 状态权重矩阵Q - 调整这些值可以改变控制器性能
        Q_lqr = np.diag([51.2938, 51.2938, 32.8281, 131.3123,  # 位置状态权重
                         51.2938, 51.2938, 131.3123, 131.3123]) # 速度状态权重

        # 控制输入权重矩阵R - 较小的rho值会使控制器更积极
        rho = 0.0005
        R_lqr = rho * np.eye(2)

        # 计算离散LQR增益
        self.K = self._dlqr(G, H, Q_lqr, R_lqr)
        print("计算得到的LQR增益矩阵 K:", self.K)
        
        # 保存系统矩阵
        self.A = A
        self.B = B
        self.G = G
        self.H = H

    def _controllability_matrix(self, A, B):
        """计算可控性矩阵"""
        n = A.shape[0]
        Tc = np.zeros((n, n*B.shape[1]))
        for i in range(n):
            Tc[:, i*B.shape[1]:(i+1)*B.shape[1]] = np.linalg.matrix_power(A, i) @ B
        return Tc

    def _dlqr(self, A, B, Q, R):
        """离散时间LQR求解器"""
        # 求解离散代数Riccati方程
        P = solve_discrete_are(A, B, Q, R)
        # 计算LQR增益
        K = inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def compute_control(self, state):
        """计算LQR控制力"""
        return -self.K @ state

    def get_system_matrices(self):
        """获取系统矩阵"""
        return self.A, self.B   #, self.G, self.H   #x k+1 =Gx k +Hu k​
