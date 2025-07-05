# ==================== 物理参数定义 ====================
class Config:
    m_car = 0.9       # 车体质量
    m_pole = 0.1      # 摆杆质量
    r_wheel = 0.0335 * 1.5  # 视觉放大后的轮子半径
    L_pole = 0.390    # 摆杆长度
    l_pole = L_pole / 2
    g = 9.8
    I_pole = (1/12) * m_pole * L_pole**2  # 摆杆惯量 
    font_name = "Songti SC"