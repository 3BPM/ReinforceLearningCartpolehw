import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from envsim.config import Config

def build_system_matrices(Ts=None):
    """
    构建系统惯性矩阵P、重力矩阵Q、状态空间A/B、离散化等，返回A, B, C, D, G, H
    """
    m_1 = Config.m_car
    m_2 = Config.m_pole
    l_1 = Config.l_body
    l_2 = Config.l_pole
    I_1 = Config.I_body
    I_2 = Config.I_pole
    r = Config.r_wheel
    g = Config.g

    # 惯性矩阵P
    P = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [(r/2)*(m_1*l_1 + m_2*l_1), (r/2)*(m_1*l_1 + m_2*l_1), m_1*l_1**2 + m_2*l_1**2 + I_1, m_2*l_1*l_2],
        [(r/2)*m_2*l_2,              (r/2)*m_2*l_2,            m_2*l_1*l_2,               m_2*l_2**2 + I_2]
    ])
    # 重力矩阵Q
    q = np.zeros((4, 10))
    q[2, 2] = (m_1*l_1 + m_2*l_1)*g
    q[3, 3] = m_2*g*l_2
    q[0, 8] = 1
    q[1, 9] = 1
    temp = inv(P) @ q
    # 连续状态空间
    A = np.block([
        [np.zeros((4, 4)), np.eye(4)],
        [temp[:, :8]]
    ])
    B = np.block([
        [np.zeros((4, 2))],
        [temp[:, 8:10]]
    ])
    C = np.block([np.eye(4), np.zeros((4, 4))])
    D = np.zeros((4, 2))
    # 离散化
    if Ts is None:
        Ts = 0.01
    G, H, _, _, _ = cont2discrete((A, B, C, D), Ts, method='zoh')
    return A, B, C, D, G, H

# ==================== LQR控制器类 ====================
class LQRController:
    def __init__(self):
        # 构建系统矩阵
        A, B, C, D, G, H = build_system_matrices()
        # 系统可控性分析
        Tc = self._controllability_matrix(G, H)
        if np.linalg.matrix_rank(Tc) == A.shape[0]:
            print(f'系统可控性分析: 系统是完全可控的(秩={np.linalg.matrix_rank(Tc)})')
        else:
            raise ValueError('系统不可控，无法设计控制器')
        # LQR控制器设计
        Q_lqr = np.diag([51.2938, 51.2938, 32.8281, 131.3123,
                         51.2938, 51.2938, 131.3123, 131.3123])
        rho = 0.0005
        R_lqr = rho * np.eye(2)
        self.K = self._dlqr(G, H, Q_lqr, R_lqr)
        print("计算得到的LQR增益矩阵 K:", self.K)
        # 保存系统矩阵
        self.A = A
        self.B = B
        self.G = G
        self.H = H

    def _controllability_matrix(self, A, B):
        n = A.shape[0]
        Tc = np.zeros((n, n*B.shape[1]))
        for i in range(n):
            Tc[:, i*B.shape[1]:(i+1)*B.shape[1]] = np.linalg.matrix_power(A, i) @ B
        return Tc

    def _dlqr(self, A, B, Q, R):
        P = solve_discrete_are(A, B, Q, R)
        K = inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def compute_control(self, state):
        return -self.K @ state

    def get_system_matrices(self):
        return self.A, self.B
