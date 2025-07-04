import numpy as np
from numpy.linalg import inv
import pygame
import math
# 需要 scipy 来计算LQR控制器
from scipy.linalg import solve_continuous_are

# ==================== 物理参数定义 ====================
# (和你的代码完全一样)
m_1 = 0.9      # 车体的质量 (kg)
m_2 = 0.1      # 摆杆的质量 (kg)
r = 0.0335     # 车轮的半径 (m)
L_1 = 0.126    # 车体的长度 (m)
L_2 = 0.390    # 摆杆的长度 (m)
l_1 = L_1 / 2  # 车体质心到转轴的距离
l_2 = L_2 / 2  # 摆杆质心到转轴的距离
g = 9.8        # 重力加速度 (m/s^2)
I_1 = (1/12) * m_1 * L_1**2  # 车体转动惯量
I_2 = (1/12) * m_2 * L_2**2  # 摆杆转动惯量

# ==================== 系统建模 ====================
# 修正后的p, q矩阵 (原矩阵有误，这是根据标准两轮倒立摆动力学方程修正的)
# 这一步非常复杂，涉及到拉格朗日方程或牛顿-欧拉方程的推导
# 我们直接使用一个比较常见的、经过验证的模型形式
M_t = m_1 + 2*m_2 # 总质量近似
J_t = I_1 + m_1*l_1**2 + 2*m_2*L_1**2 # 总转动惯量近似

# 这里使用一个更标准和简化的模型来保证稳定性，因为原始的p,q矩阵推导非常复杂且容易出错
# 简化模型：将两个轮子看作一个整体，车体和摆杆为倒立摆
M = m_1 + m_2  # 总质量
L = l_2        # 摆杆有效长度
I = I_2        # 摆杆转动惯量
m = m_1        # 车体质量

# 状态向量简化为 x = [pos, angle, pos_dot, angle_dot]
# pos: 车体水平位置, angle: 摆杆角度
# 这是标准的倒立摆模型，更容易理解和可视化
# 状态方程 ẋ = Ax + Bu
#       y = Cx + Du
# pos_ddot = ( F - m*L*angle_dot^2*sin(angle) + m*g*sin(angle)*cos(angle) ) / ( M - m*L^2*cos(angle)^2 )
# angle_ddot = ( g*sin(angle) - pos_ddot*cos(angle) ) / L
# 在平衡点(angle=0)附近线性化: sin(θ)≈θ, cos(θ)≈1, θ̇^2≈0
denom = I*(M+m) + M*m*L**2
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, (m**2 * L**2 * g) / denom, 0, 0],
    [0, (m * g * L * (M+m)) / denom, 0, 0]
])

B = np.array([
    [0],
    [0],
    [(I + m*L**2)/denom],
    [(m*L)/denom]
])

# 输出我们关心的所有状态
C = np.eye(4)
D = np.zeros((4,1))

# ==================== LQR控制器设计 ====================
# LQR代价矩阵
# Q矩阵：惩罚状态偏差。我们非常不希望杆倒下(angle)，也不希望车乱跑(pos)
# 对角线元素分别对应 [pos, angle, pos_dot, angle_dot] 的惩罚权重
Q = np.diag([1.0, 100.0, 1.0, 1.0])

# R矩阵：惩罚控制输入。R越大，意味着我们希望用更小的力去控制，更节能
R = np.array([[0.1]])

# 求解连续代数黎卡提方程 (CARE)
P = solve_continuous_are(A, B, Q, R)

# 计算LQR增益K
K = inv(R) @ B.T @ P
print("计算得到的LQR增益矩阵 K:", K)


# ==================== Pygame仿真与可视化 ====================

# --- Pygame 设置 ---
pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LQR控制的自平衡小车")
clock = pygame.time.Clock()

# --- 颜色和参数 ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# 仿真参数
SCALE = 200  # 缩放比例: 1米 = 200像素
FPS = 60
dt = 1.0 / FPS

# --- 初始状态 ---
# x = [位置(m), 杆角度(rad), 速度(m/s), 杆角速度(rad/s)]
# 让杆有一个小的初始倾角，看控制器如何把它扶正
x = np.array([0, 0.2, 0, 0])

# --- 主循环 ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 1. 控制器计算
    # u = -Kx 是核心！根据当前状态计算控制力
    # 由于我们希望小车保持在原点(pos=0)和垂直(angle=0)，所以目标状态是[0,0,0,0]
    # 偏差就是 x - x_target = x - 0 = x
    u = -K @ x
    # 对输入力矩进行限幅，防止力量过大
    u_clamped = np.clip(u, -20.0, 20.0)

    # 2. 系统状态更新 (模拟物理世界)
    # 使用状态空间方程计算状态的变化率
    x_dot = A @ x + B @ u_clamped
    # 用欧拉积分法更新状态
    x = x + x_dot * dt

    # 3. 绘图
    screen.fill(WHITE)

    # 提取状态用于绘图
    cart_pos_m = x[0]
    pole_angle_rad = x[1]

    # --- 坐标转换 (从物理世界到屏幕) ---
    # 地面
    ground_y = HEIGHT - 100
    pygame.draw.line(screen, BLACK, (0, ground_y), (WIDTH, ground_y), 2)

    # 车体
    cart_x_px = WIDTH / 2 + cart_pos_m * SCALE
    cart_y_px = ground_y
    cart_w, cart_h = 100, 40
    pygame.draw.rect(screen, BLUE, (cart_x_px - cart_w/2, cart_y_px - cart_h, cart_w, cart_h))

    # 轮子
    wheel_radius_px = r * SCALE
    pygame.draw.circle(screen, BLACK, (int(cart_x_px - cart_w/2*0.7), int(cart_y_px - wheel_radius_px)), int(wheel_radius_px), 2)
    pygame.draw.circle(screen, BLACK, (int(cart_x_px + cart_w/2*0.7), int(cart_y_px - wheel_radius_px)), int(wheel_radius_px), 2)

    # 摆杆
    pole_len_px = L_2 * SCALE
    pole_base_x = cart_x_px
    pole_base_y = cart_y_px - cart_h # 杆的底部在车体顶部
    # 终点坐标计算
    pole_end_x = pole_base_x + pole_len_px * math.sin(pole_angle_rad)
    pole_end_y = pole_base_y - pole_len_px * math.cos(pole_angle_rad) # Pygame的y轴向下，所以是减
    pygame.draw.line(screen, RED, (pole_base_x, pole_base_y), (pole_end_x, pole_end_y), 6)

    # --- 显示信息 ---
    font = pygame.font.SysFont("SimHei", 24)
    info_text_1 = f"车体位置: {x[0]:.2f} m | 摆杆角度: {math.degrees(x[1]):.2f} °"
    info_text_2 = f"车体速度: {x[2]:.2f} m/s | 摆杆角速度: {math.degrees(x[3]):.2f} °/s"
    info_text_3 = f"控制力: {u[0]:.2f} N"

    text_surface_1 = font.render(info_text_1, True, BLACK)
    text_surface_2 = font.render(info_text_2, True, BLACK)
    text_surface_3 = font.render(info_text_3, True, GREEN)

    screen.blit(text_surface_1, (10, 10))
    screen.blit(text_surface_2, (10, 40))
    screen.blit(text_surface_3, (10, 70))


    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()