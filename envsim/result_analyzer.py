import numpy as np
from envsim.config import Config

import matplotlib
matplotlib.rcParams['font.sans-serif'] = [Config.font_name]
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import math

class ResultAnalyzer:
    """仿真结果分析器"""

    def __init__(self):
        self.time_history = []
        self.state_history = []
        self.control_history = []
        self.performance_metrics = {}

    def add_data_point(self, time, state, control_force):
        """添加数据点"""
        self.time_history.append(time)
        self.state_history.append(state.copy())
        self.control_history.append(control_force)

    def clear_data(self):
        """清空数据"""
        self.time_history = []
        self.state_history = []
        self.control_history = []
        self.performance_metrics = {}

    def calculate_performance_metrics(self):
        """计算性能指标"""
        if not self.state_history:
            return {}

        states = np.array(self.state_history)
        controls = np.array(self.control_history)
        times = np.array(self.time_history)

        # 计算各项指标
        metrics = {}

        # 1. 最大偏差
        metrics['max_position_error'] = np.max(np.abs(states[:, 0]))
        metrics['max_angle_error'] = np.max(np.abs(states[:, 1]))
        metrics['max_velocity_error'] = np.max(np.abs(states[:, 2]))
        metrics['max_angular_velocity_error'] = np.max(np.abs(states[:, 3]))

        # 2. 稳态误差（最后1秒的平均值）
        final_second_mask = times >= (times[-1] - 1.0)
        if np.any(final_second_mask):
            final_states = states[final_second_mask]
            metrics['steady_state_position'] = np.mean(np.abs(final_states[:, 0]))
            metrics['steady_state_angle'] = np.mean(np.abs(final_states[:, 1]))
            metrics['steady_state_velocity'] = np.mean(np.abs(final_states[:, 2]))
            metrics['steady_state_angular_velocity'] = np.mean(np.abs(final_states[:, 3]))

        # 3. 控制性能
        metrics['max_control_force'] = np.max(np.abs(controls))
        metrics['avg_control_force'] = np.mean(np.abs(controls))
        metrics['control_energy'] = np.sum(controls**2) * Config.dt

        # 4. 响应时间（达到稳态的时间）
        threshold = 0.05  # 5%的稳态值作为阈值
        metrics['settling_time'] = self._calculate_settling_time(states, times, threshold)

        # 5. 超调量
        metrics['overshoot_angle'] = self._calculate_overshoot(states[:, 1])

        self.performance_metrics = metrics
        return metrics

    def _calculate_settling_time(self, states, times, threshold):
        """计算调节时间"""
        angle_abs = np.abs(states[:, 1])
        steady_state = np.mean(angle_abs[-int(1.0/Config.dt):])  # 最后1秒的平均值

        # 找到首次进入稳态的时间
        settled_mask = angle_abs <= (steady_state + threshold)
        settled_indices = np.where(settled_mask)[0]

        if len(settled_indices) > 0:
            # 检查是否持续稳定
            for i in settled_indices:
                if i + int(0.5/Config.dt) < len(angle_abs):  # 检查后续0.5秒
                    if np.all(angle_abs[i:i+int(0.5/Config.dt)] <= (steady_state + threshold)):
                        return times[i]

        return times[-1]  # 如果没有达到稳态，返回总时间

    def _calculate_overshoot(self, angle_history):
        """计算超调量"""
        max_angle = np.max(np.abs(angle_history))
        steady_state = np.mean(np.abs(angle_history[-int(1.0/Config.dt):]))

        if steady_state > 0:
            return (max_angle - steady_state) / steady_state * 100
        return 0

    def plot_detailed_response(self, save_path=None):
        """绘制详细的响应曲线"""
        if not self.state_history:
            print("没有数据可供分析")
            return

        # 计算性能指标
        self.calculate_performance_metrics()

        # 创建图形
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle('独轮自平衡车LQR控制响应分析', fontsize=16, fontweight='bold')

        # 使用GridSpec创建子图布局
        gs = GridSpec(4, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1, 1, 1])

        times = np.array(self.time_history)
        states = np.array(self.state_history)
        controls = np.array(self.control_history)

        # 1. 轮轴位置响应
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, states[:, 0], 'b', linewidth=1.5, label='位置响应')
        max_val = np.max(np.abs(states[:, 0]))
        max_idx = np.argmax(np.abs(states[:, 0]))
        ax1.axvline(x=times[max_idx], color='r', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('轮轴位置响应')
        ax1.set_ylabel('位置 (m)')
        ax1.legend([f'最大值: {max_val:.3f}m'])
        ax1.set_xlim([0, min(7, times[-1])])

        # 2. 摆杆角度响应
        ax2 = fig.add_subplot(gs[1, 0])
        angle_deg = np.rad2deg(states[:, 1])
        ax2.plot(times, angle_deg, 'b', linewidth=1.5, label='角度响应')
        max_val = np.max(np.abs(angle_deg))
        max_idx = np.argmax(np.abs(angle_deg))
        ax2.axvline(x=times[max_idx], color='r', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('摆杆角度响应')
        ax2.set_ylabel('角度 (°)')
        ax2.legend([f'最大值: {max_val:.2f}°'])
        ax2.set_xlim([0, min(7, times[-1])])

        # 3. 轮轴速度响应
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(times, states[:, 2], 'b', linewidth=1.5, label='速度响应')
        max_val = np.max(np.abs(states[:, 2]))
        max_idx = np.argmax(np.abs(states[:, 2]))
        ax3.axvline(x=times[max_idx], color='r', linestyle='--', alpha=0.7)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('轮轴速度响应')
        ax3.set_ylabel('速度 (m/s)')
        ax3.legend([f'最大值: {max_val:.3f}m/s'])
        ax3.set_xlim([0, min(7, times[-1])])

        # 4. 角速度响应
        ax4 = fig.add_subplot(gs[3, 0])
        angular_velocity_deg = np.rad2deg(states[:, 3])
        ax4.plot(times, angular_velocity_deg, 'b', linewidth=1.5, label='角速度响应')
        max_val = np.max(np.abs(angular_velocity_deg))
        max_idx = np.argmax(np.abs(angular_velocity_deg))
        ax4.axvline(x=times[max_idx], color='r', linestyle='--', alpha=0.7)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('角速度响应')
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('角速度 (°/s)')
        ax4.legend([f'最大值: {max_val:.2f}°/s'])
        ax4.set_xlim([0, min(7, times[-1])])

        # 5. 控制力
        ax5 = fig.add_subplot(gs[0, 1])
        ax5.plot(times, controls, 'g', linewidth=1.5)
        ax5.grid(True, alpha=0.3)
        ax5.set_title('控制力')
        ax5.set_ylabel('力 (N)')
        ax5.set_xlim([0, min(7, times[-1])])

        # 6. 性能指标表格
        ax6 = fig.add_subplot(gs[1:, 1])
        ax6.axis('off')

        # 创建性能指标表格
        metrics_text = "性能指标:\n\n"
        metrics_text += f"最大角度偏差: {self.performance_metrics.get('max_angle_error', 0):.3f} rad\n"
        metrics_text += f"稳态角度误差: {self.performance_metrics.get('steady_state_angle', 0):.3f} rad\n"
        metrics_text += f"调节时间: {self.performance_metrics.get('settling_time', 0):.2f} s\n"
        metrics_text += f"超调量: {self.performance_metrics.get('overshoot_angle', 0):.1f}%\n"
        metrics_text += f"最大控制力: {self.performance_metrics.get('max_control_force', 0):.2f} N\n"
        metrics_text += f"平均控制力: {self.performance_metrics.get('avg_control_force', 0):.2f} N\n"
        metrics_text += f"控制能量: {self.performance_metrics.get('control_energy', 0):.2f} J\n"

        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        plt.show()

    def plot_phase_portrait(self, save_path=None):
        """绘制相图"""
        if not self.state_history:
            print("没有数据可供分析")
            return

        states = np.array(self.state_history)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('相图分析', fontsize=16, fontweight='bold')

        # 位置-速度相图
        ax1.plot(states[:, 0], states[:, 2], 'b-', linewidth=1)
        ax1.scatter(states[0, 0], states[0, 2], c='red', s=100, marker='o', label='起始点')
        ax1.scatter(states[-1, 0], states[-1, 2], c='green', s=100, marker='s', label='结束点')
        ax1.set_xlabel('位置 (m)')
        ax1.set_ylabel('速度 (m/s)')
        ax1.set_title('位置-速度相图')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 角度-角速度相图
        ax2.plot(states[:, 1], states[:, 3], 'b-', linewidth=1)
        ax2.scatter(states[0, 1], states[0, 3], c='red', s=100, marker='o', label='起始点')
        ax2.scatter(states[-1, 1], states[-1, 3], c='green', s=100, marker='s', label='结束点')
        ax2.set_xlabel('角度 (rad)')
        ax2.set_ylabel('角速度 (rad/s)')
        ax2.set_title('角度-角速度相图')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 位置-角度相图
        ax3.plot(states[:, 0], states[:, 1], 'b-', linewidth=1)
        ax3.scatter(states[0, 0], states[0, 1], c='red', s=100, marker='o', label='起始点')
        ax3.scatter(states[-1, 0], states[-1, 1], c='green', s=100, marker='s', label='结束点')
        ax3.set_xlabel('位置 (m)')
        ax3.set_ylabel('角度 (rad)')
        ax3.set_title('位置-角度相图')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 速度-角速度相图
        ax4.plot(states[:, 2], states[:, 3], 'b-', linewidth=1)
        ax4.scatter(states[0, 2], states[0, 3], c='red', s=100, marker='o', label='起始点')
        ax4.scatter(states[-1, 2], states[-1, 3], c='green', s=100, marker='s', label='结束点')
        ax4.set_xlabel('速度 (m/s)')
        ax4.set_ylabel('角速度 (rad/s)')
        ax4.set_title('速度-角速度相图')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相图已保存到: {save_path}")

        plt.show()

    def generate_report(self, save_path=None):
        """生成分析报告"""
        if not self.state_history:
            print("没有数据可供分析")
            return

        self.calculate_performance_metrics()

        report = f"""
独轮自平衡车LQR控制仿真分析报告
=====================================

仿真参数:
- 仿真时长: {self.time_history[-1]:.2f} 秒
- 时间步长: {Config.dt:.4f} 秒
- 数据点数: {len(self.state_history)}

性能指标:
- 最大角度偏差: {self.performance_metrics.get('max_angle_error', 0):.4f} rad ({np.rad2deg(self.performance_metrics.get('max_angle_error', 0)):.2f}°)
- 稳态角度误差: {self.performance_metrics.get('steady_state_angle', 0):.4f} rad ({np.rad2deg(self.performance_metrics.get('steady_state_angle', 0)):.2f}°)
- 调节时间: {self.performance_metrics.get('settling_time', 0):.2f} 秒
- 超调量: {self.performance_metrics.get('overshoot_angle', 0):.1f}%
- 最大控制力: {self.performance_metrics.get('max_control_force', 0):.2f} N
- 平均控制力: {self.performance_metrics.get('avg_control_force', 0):.2f} N
- 控制能量: {self.performance_metrics.get('control_energy', 0):.2f} J

控制效果评估:
"""

        # 评估控制效果
        max_angle = self.performance_metrics.get('max_angle_error', 0)
        settling_time = self.performance_metrics.get('settling_time', 0)
        overshoot = self.performance_metrics.get('overshoot_angle', 0)

        if max_angle < 0.1 and settling_time < 3.0 and overshoot < 20:
            report += "- 控制效果: 优秀\n"
        elif max_angle < 0.2 and settling_time < 5.0 and overshoot < 30:
            report += "- 控制效果: 良好\n"
        elif max_angle < 0.3 and settling_time < 7.0 and overshoot < 50:
            report += "- 控制效果: 一般\n"
        else:
            report += "- 控制效果: 需要改进\n"

        report += f"""
建议:
- 如需提高响应速度，可增加控制权重
- 如需减少超调，可调整状态权重矩阵
- 如需提高稳态精度，可增加积分控制
"""

        print(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"报告已保存到: {save_path}")

        return report