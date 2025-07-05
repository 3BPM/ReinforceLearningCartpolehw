import numpy as np
from config import Config

# ==================== 仿真器类 ====================
class UnicycleSimulator:
    def __init__(self, controller, result_analyzer=None,state=None):
        self.controller = controller
        self.A, self.B = controller.get_system_matrices()
        self.state = np.array(state) if state is not None else np.array(Config.initial_state)

        """[ θL    ]  (左轮角度)
      [ θR    ]  (右轮角度)
      [ θ1    ]  (连杆1角度)
      [ θ2    ]  (连杆2角度)
state  =  [ θL_dot]  (左轮角速度)
      [ θR_dot]  (右轮角速度)
      [ θ1_dot]  (连杆1角速度)
      [ θ2_dot]  (连杆2角速度)"""
        self.dt = Config.dt
        self.speed_multiplier = 1.0
        self.is_lqr_active = True
        self.apply_manual_force = True
        self.manual_force = 0.0

        # 结果分析器
        self.result_analyzer = result_analyzer
        self.simulation_time = 0.0
        self.is_recording = False

    def reset(self):
        """重置仿真状态"""
        self.state = np.array(Config.initial_state)
        self.manual_force = 0.0
        self.is_lqr_active = True
        self.simulation_time = 0.0

        # 清空分析器数据
        if self.result_analyzer:
            self.result_analyzer.clear_data()

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

    def start_recording(self):
        """开始记录数据"""
        self.is_recording = True
        if self.result_analyzer:
            self.result_analyzer.clear_data()

    def stop_recording(self):
        """停止记录数据"""
        self.is_recording = False

    def step(self):
        """执行一步仿真"""
        # 计算LQR控制力
        u_lqr = self.controller.compute_control(self.state) if self.is_lqr_active else np.array([0.0])

        # 计算总控制力
        # # u_lqr 是 shape=(2,) 的控制向量，分别对应左右轮的控制力
        # manual_force 是人为施加的相同干预力，需要同时作用在左右通道
        # total_force 是最终的控制输出，包含自动控制和人为干预
        if self.apply_manual_force:
            total_force = u_lqr + np.array([self.manual_force, self.manual_force])
        else:
            total_force = u_lqr
        force_clamped = np.clip(np.array([[total_force]]), -50.0, 50.0)
        force_clamped = force_clamped.squeeze()  # 从 (1,1,2) 变为 (2,)
        # 更新状态
        x_dot = self.A @ self.state + (self.B @ force_clamped).flatten()
        effective_dt = self.dt * self.speed_multiplier
        self.state = self.state + x_dot * effective_dt

        # 更新时间
        self.simulation_time += effective_dt

        # 记录数据
        if self.is_recording and self.result_analyzer:
            self.result_analyzer.add_data_point(
                self.simulation_time,
                self.state,
                force_clamped[0, 0]
            )
        if self.state[2] < -np.pi or self.state[2] > np.pi or self.state[3] < -np.pi or self.state[3] > np.pi:
            print("摆杆角度超出范围，仿真结束")
            return True
        else:
            return False

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

    def get_simulation_time(self):
        """获取仿真时间"""
        return self.simulation_time