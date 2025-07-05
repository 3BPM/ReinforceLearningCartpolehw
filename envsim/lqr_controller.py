import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_continuous_are
from config import Config

# ==================== LQR控制器类 ====================
class LQRController:
    def __init__(self):
        # 状态空间模型
        M = Config.m_car + Config.m_pole
        m = Config.m_pole
        L = Config.l_pole
        I = Config.I_pole
        denom = I * (M) + m * M * L**2

        self.A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, (m**2 * L**2 * Config.g) / denom, 0, 0],
            [0, (m * Config.g * L * M) / denom, 0, 0]
        ])

        self.B = np.array([
            [0],
            [0],
            [(I + m * L**2) / denom],
            [(m * L) / denom]
        ])

        # LQR控制器设计
        Q = np.diag([1.0, 100.0, 1.0, 1.0])
        R = np.array([[0.1]])
        P = solve_continuous_are(self.A, self.B, Q, R)
        self.K = inv(R) @ self.B.T @ P
        print("计算得到的LQR增益矩阵 K:", self.K)

    def compute_control(self, state):
        """计算LQR控制力"""
        return -self.K @ state

    def get_system_matrices(self):
        """获取系统矩阵"""
        return self.A, self.B 