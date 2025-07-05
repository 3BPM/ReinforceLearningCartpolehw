# ==================== 配置参数定义 ====================
class Config:
    # 物理参数
    m_car = 0.9       # 车体质量
    m_pole = 0.1      # 摆杆质量
    r_wheel = 0.0335 * 1.5  # 视觉放大后的轮子半径
    L_pole = 0.390    # 摆杆长度
    l_pole = L_pole / 2
    g = 9.8
    I_pole = (1/12) * m_pole * L_pole**2  # 摆杆惯量
    
    # 仿真参数
    dt = 1.0 / 60.0   # 时间步长
    max_time = 10.0   # 最大仿真时间
    
    # 控制参数
    Q = [1.0, 100.0, 1.0, 1.0]  # LQR权重矩阵对角线元素
    R = 0.1                     # LQR控制权重
    
    # 可视化参数
    window_width = 1200
    window_height = 700
    scale = 250
    fps = 60

    font_name = "Songti SC"