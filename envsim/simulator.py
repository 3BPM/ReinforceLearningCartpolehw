import numpy as np

# ==================== 仿真器类 ====================
class UnicycleSimulator:
    def __init__(self, controller):
        self.controller = controller
        self.A, self.B = controller.get_system_matrices()
        self.state = np.array([0, 0.2, 0, 0])  # 初始状态
        self.dt = 1.0 / 60.0
        self.speed_multiplier = 1.0
        self.is_lqr_active = True
        self.apply_manual_force = True
        self.manual_force = 0.0

    def reset(self):
        """重置仿真状态"""
        self.state = np.array([0, 0.2, 0, 0])
        self.manual_force = 0.0
        self.is_lqr_active = True

    def set_manual_force(self, force):
        """设置手动施力"""
        self.manual_force = force

    def set_lqr_active(self, active):
        """设置LQR控制器是否激活"""
        self.is_lqr_active = active

    def set_apply_manual_force(self, apply):
        """设置是否应用手动施力"""
        self.apply_manual_force = apply

    def set_speed_multiplier(self, multiplier):
        """设置仿真速度倍数"""
        self.speed_multiplier = max(0.1, multiplier)

    def step(self):
        """执行一步仿真"""
        # 计算LQR控制力
        u_lqr = self.controller.compute_control(self.state) if self.is_lqr_active else np.array([0.0])
        
        # 计算总控制力
        total_force = u_lqr[0] + self.manual_force if self.apply_manual_force else u_lqr[0]
        force_clamped = np.clip(np.array([[total_force]]), -50.0, 50.0)
        
        # 更新状态
        x_dot = self.A @ self.state + (self.B @ force_clamped).flatten()
        effective_dt = self.dt * self.speed_multiplier
        self.state = self.state + x_dot * effective_dt

    def get_state(self):
        """获取当前状态"""
        return self.state.copy()

    def get_control_info(self):
        """获取控制信息"""
        u_lqr = self.controller.compute_control(self.state) if self.is_lqr_active else np.array([0.0])
        return {
            'lqr_force': u_lqr[0],
            'manual_force': self.manual_force,
            'is_lqr_active': self.is_lqr_active,
            'apply_manual_force': self.apply_manual_force
        } 