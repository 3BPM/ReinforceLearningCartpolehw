#!/usr/bin/env python3
"""
独轮自平衡车仿真结果分析示例
演示如何使用ResultAnalyzer进行离线分析
"""

import numpy as np
import matplotlib.pyplot as plt
from envsim.result_analyzer import ResultAnalyzer
from envsim.config import Config

def generate_sample_data():
    """生成示例数据用于演示"""
    analyzer = ResultAnalyzer()
    
    # 模拟仿真数据
    t = np.linspace(0, 10, 1000)  # 10秒，1000个数据点
    dt = t[1] - t[0]
    
    # 模拟状态响应（带噪声的衰减振荡）
    omega = 2.0  # 自然频率
    zeta = 0.3   # 阻尼比
    
    # 角度响应（主要关心的状态）
    angle = 0.2 * np.exp(-zeta * omega * t) * np.cos(omega * np.sqrt(1 - zeta**2) * t)
    angle += 0.01 * np.random.randn(len(t))  # 添加噪声
    
    # 角速度
    angular_velocity = np.gradient(angle, dt)
    
    # 位置（积分得到）
    position = np.cumsum(angular_velocity * Config.r_wheel) * dt
    
    # 速度
    velocity = np.gradient(position, dt)
    
    # 控制力（基于LQR控制律）
    K = np.array([[-1.0, -10.0, -0.5, -1.0]])  # 假设的控制增益
    control_force = np.zeros(len(t))
    
    for i in range(len(t)):
        state = np.array([position[i], angle[i], velocity[i], angular_velocity[i]])
        control_force[i] = -K @ state
    
    # 添加数据到分析器
    for i in range(len(t)):
        state = np.array([position[i], angle[i], velocity[i], angular_velocity[i]])
        analyzer.add_data_point(t[i], state, control_force[i])
    
    return analyzer

def main():
    """主函数"""
    print("=== 独轮自平衡车仿真结果分析示例 ===")
    
    # 生成示例数据
    print("正在生成示例数据...")
    analyzer = generate_sample_data()
    
    # 计算性能指标
    print("正在计算性能指标...")
    metrics = analyzer.calculate_performance_metrics()
    
    print("\n性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 生成分析报告
    print("\n正在生成分析报告...")
    analyzer.generate_report("example_report.txt")
    
    # 绘制详细响应曲线
    print("正在绘制响应曲线...")
    analyzer.plot_detailed_response("example_response.png")
    
    # 绘制相图
    print("正在绘制相图...")
    analyzer.plot_phase_portrait("example_phase.png")
    
    print("\n分析完成！生成的文件:")
    print("- example_report.txt: 文本报告")
    print("- example_response.png: 响应曲线图")
    print("- example_phase.png: 相图")
    
    # 显示一些关键指标
    print(f"\n关键性能指标:")
    print(f"- 最大角度偏差: {np.rad2deg(metrics['max_angle_error']):.2f}°")
    print(f"- 调节时间: {metrics['settling_time']:.2f} 秒")
    print(f"- 超调量: {metrics['overshoot_angle']:.1f}%")
    print(f"- 控制能量: {metrics['control_energy']:.2f} J")

if __name__ == "__main__":
    main() 