import numpy as np
import pygame
import sys
from numpy.linalg import inv
import control as ct

# 初始化Pygame
pygame.init()

# 屏幕尺寸
width, height = 1000, 700
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Cart-Pendulum System Simulation (State Space)")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
GREEN = (0, 255, 0)

# ==================== 物理参数定义 ====================
m_1 = 0.9      # 车体的质量(kg)
m_2 = 0.1      # 摆杆的质量(kg)
r = 0.0335     # 车轮的半径(m)
L_1 = 0.126    # 车体的长度(m)
L_2 = 0.390    # 摆杆的长度(m)
l_1 = L_1 / 2  # 车体质心到转轴的距离(m)
l_2 = L_2 / 2  # 摆杆质心到转轴的距离(m)
g = 9.8        # 重力加速度(m/s^2)

# 计算转动惯量
I_1 = (1/12) * m_1 * L_1**2  # 车体转动惯量(kg·m^2)
I_2 = (1/12) * m_2 * L_2**2  # 摆杆转动惯量(kg·m^2)

# ==================== 系统建模 ====================
# 构建惯性矩阵 p (4x4矩阵)
p = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [(r/2)*(m_1*l_1 + m_2*L_1), (r/2)*(m_1*l_1 + m_2*L_1),
     m_1*l_1**2 + m_2*L_1**2 + I_1, m_2*L_1*l_2],
    [(r/2)*m_2*l_2, (r/2)*m_2*l_2, m_2*L_1*l_2, m_2*l_2**2 + I_2]
])

# 构建重力/外力矩阵 q (4x10矩阵)
q = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, (m_1*l_1 + m_2*L_1)*g, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, m_2*g*l_2, 0, 0, 0, 0, 0, 0]
])

# 计算临时矩阵 (p的逆乘以q)
temp = inv(p) @ q

# ==================== 状态空间模型 ====================
# 构建A矩阵 (8x8状态矩阵)
A = np.vstack([
    np.hstack([np.zeros((4,4)), np.eye(4)]),  # 前4行
    temp[:, :8]                                # 后4行
])

# 构建B矩阵 (8x2控制输入矩阵)
B = np.vstack([
    np.zeros((4,2)),  # 前4行
    temp[:, 8:]       # 后4行
])

# 构建C矩阵 (4x8输出矩阵)
C = np.hstack([np.eye(4), np.zeros((4,4))])

# 构建D矩阵 (4x2前馈矩阵)
D = np.zeros((4,2))

# 使用control库创建状态空间系统
sys_ss = ct.ss(A, B, C, D)

# ==================== 模拟参数 ====================
dt = 0.01      # 时间步长(s)
sim_time = 10  # 总模拟时间(s)
num_steps = int(sim_time / dt)

# 初始状态 [θ_L, θ_R, θ₁, θ₂, θ̇_L, θ̇_R, θ̇₁, θ̇₂]
x = np.array([[0], [0], [np.pi/6], [0], [0], [0], [0], [0]])  # 初始小角度倾斜

# 控制输入 [u_L, u_R]
u = np.array([[0], [0]])

# ==================== Pygame可视化 ====================
# 缩放因子
scale = 200  # 像素/米
cart_width_px = int(L_1 * scale)
cart_height_px = int(L_1/2 * scale)
pendulum_length_px = int(L_2 * scale)
wheel_radius_px = int(r * scale)

# 轨道位置
track_y = height // 2 + 100

# 字体
font = pygame.font.SysFont('SimHei', 16)

def draw_system(state):
    # 计算小车位置
    cart_center_x = width // 2 + (state[0,0] + state[1,0])/2 * r * scale
    cart_center_y = track_y

    # 绘制轨道
    pygame.draw.line(screen, BLACK, (50, track_y), (width-50, track_y), 2)

    # 绘制车轮
    wheel_L_x = cart_center_x - cart_width_px//2
    wheel_R_x = cart_center_x + cart_width_px//2
    pygame.draw.circle(screen, BLACK, (int(wheel_L_x), int(cart_center_y + cart_height_px//2)), wheel_radius_px, 2)
    pygame.draw.circle(screen, BLACK, (int(wheel_R_x), int(cart_center_y + cart_height_px//2)), wheel_radius_px, 2)

    # 绘制小车主体
    cart_rect = pygame.Rect(
        int(cart_center_x - cart_width_px//2),
        int(cart_center_y - cart_height_px//2),
        cart_width_px,
        cart_height_px
    )
    pygame.draw.rect(screen, BLUE, cart_rect)

    # 计算摆杆端点
    pendulum_x = cart_center_x + pendulum_length_px * np.sin(state[2,0])
    pendulum_y = cart_center_y - cart_height_px//2 - pendulum_length_px * np.cos(state[2,0])

    # 绘制摆杆
    pygame.draw.line(
        screen, RED,
        (cart_center_x, cart_center_y - cart_height_px//2),
        (pendulum_x, pendulum_y),
        5
    )

    # 绘制摆球
    pygame.draw.circle(screen, RED, (int(pendulum_x), int(pendulum_y)), 10)

    # 显示状态信息
    info_text = [
        f"θ₁ (车体角度): {state[2,0]:.2f} rad",
        f"θ₂ (摆杆角度): {state[3,0]:.2f} rad",
        f"θ̇₁ (车体角速度): {state[6,0]:.2f} rad/s",
        f"θ̇₂ (摆杆角速度): {state[7,0]:.2f} rad/s",
        f"系统类型: {sys_ss.__class__.__name__}"
    ]

    for i, text in enumerate(info_text):
        text_surface = font.render(text, True, BLACK)
        screen.blit(text_surface, (20, 20 + i * 25))

# ==================== 主模拟循环 ====================
clock = pygame.time.Clock()
running = True
step = 0
paused = False
sim_speed = 1.0  # 模拟速度因子

while running and step < num_steps:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused  # 空格键切换暂停/继续
            elif event.key == pygame.K_UP:
                sim_speed = min(sim_speed + 0.5, 5.0)  # 增加模拟速度
            elif event.key == pygame.K_DOWN:
                sim_speed = max(sim_speed - 0.5, 0.1)  # 减少模拟速度

    if not paused:
    # 简单控制输入 (示例: 按键控制)
    keys = pygame.key.get_pressed()
    u = np.array([[0.0], [0.0]])  # 重置控制输入

    if keys[pygame.K_LEFT]:
        u[0,0] = -0.1  # 左轮扭矩
    if keys[pygame.K_RIGHT]:
        u[1,0] = 0.1   # 右轮扭矩

    # 状态更新
    x_dot = A @ x + B @ u
        x = x + x_dot * dt * sim_speed
        step += 1

    # 清屏
    screen.fill(WHITE)

    # 绘制系统
    draw_system(x)

    # 显示控制信息
    control_text = [
        "使用左右方向键施加扭矩",
        "空格键: 暂停/继续",
        f"模拟速度: {sim_speed:.1f}x (上下方向键调整)",
        "暂停中..." if paused else "运行中",
        f"使用 control.ss 创建的状态空间系统"
    ]

    for i, text in enumerate(control_text):
        text_surface = font.render(text, True, GREEN if i == 3 else BLACK)
        screen.blit(text_surface, (width - 300, 20 + i * 25))

    # 更新屏幕
    pygame.display.flip()

    # 控制帧率
    clock.tick(60)

pygame.quit()
sys.exit()