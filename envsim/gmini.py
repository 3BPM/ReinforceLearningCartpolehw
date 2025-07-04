import numpy as np
from numpy.linalg import inv
import pygame
import math
from scipy.linalg import solve_continuous_are

# ==================== 物理参数定义 (简化模型) ====================
m_car = 0.9      # 车体的质量 (kg)
m_pole = 0.1     # 摆杆的质量 (kg)
r_wheel = 0.0335 # 车轮的半径 (m) (用于可视化)
L_pole = 0.390   # 摆杆的长度 (m)
l_pole = L_pole / 2 # 摆杆质心到转轴的距离
g = 9.8        # 重力加速度 (m/s^2)
I_pole = (1/12) * m_pole * L_pole**2  # 摆杆转动惯量

# ==================== 状态空间模型 (标准倒立摆线性化模型) ====================
# 状态向量 x = [pos, angle, pos_dot, angle_dot]ᵀ
# pos: 车体水平位置, angle: 摆杆角度(偏离垂直)
M = m_car + m_pole
m = m_pole
L = l_pole
I = I_pole

# 在平衡点(angle=0)附近线性化后的A, B矩阵
denom = I * (M) + m * M * L**2
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, (m**2 * L**2 * g) / denom, 0, 0],
    [0, (m * g * L * M) / denom, 0, 0]
])

B = np.array([
    [0],
    [0],
    [(I + m*L**2)/denom],
    [(m*L)/denom]
])

# ==================== LQR控制器设计 ====================
# Q矩阵: 惩罚状态偏差。我们非常不希望杆倒下(angle)，也不希望车乱跑(pos)
Q = np.diag([1.0, 100.0, 1.0, 1.0]) # [pos, angle, pos_dot, angle_dot]
# R矩阵: 惩罚控制输入。R越大，意味着我们希望用更小的力去控制，更节能
R = np.array([[0.1]])

# 求解连续代数黎卡提方程 (CARE)
P = solve_continuous_are(A, B, Q, R)
# 计算LQR增益K
K = inv(R) @ B.T @ P
print("计算得到的LQR增益矩阵 K:", K)


# ==================== Pygame仿真与可视化 ====================

# --- Pygame 设置 ---
pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LQR控制的自平衡小车 (空格:暂停, ↑/↓:速度, 滑块:施力, F:施力开关, L:LQR开关)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("SimHei", 24)

# --- 颜色和参数 ---
C_BLACK = (0, 0, 0)
C_WHITE = (255, 255, 255)
C_RED = (220, 50, 50)
C_BLUE = (50, 100, 200)
C_GREEN = (50, 200, 100)
C_GREY = (200, 200, 200)
C_DARK_GREY = (100, 100, 100)

# --- 仿真控制参数 ---
SCALE = 250      # 缩放比例: 1米 = 250像素
FPS = 60
dt = 1.0 / FPS
is_paused = False
speed_multiplier = 1.0
apply_manual_force = True  # 是否应用手动施力
is_lqr_active = True       # 新增: LQR控制器是否激活

# --- 初始状态 ---
# x = [位置(m), 杆角度(rad), 速度(m/s), 杆角速度(rad/s)]
x = np.array([0, 0.2, 0, 0]) # 初始给一个倾角

# --- 滑块参数 ---
slider_max_force = 30.0  # 牛顿
manual_force = 0.0
slider_rect = pygame.Rect(WIDTH // 4, HEIGHT - 50, WIDTH // 2, 20)
handle_rect = pygame.Rect(slider_rect.centerx - 10, slider_rect.y - 5, 20, 30)
is_dragging = False

# --- 主循环 ---
running = True
while running:
    # --- 1. 事件处理 (键盘和鼠标) ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # 键盘事件
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_paused = not is_paused
            elif event.key == pygame.K_UP or event.key == pygame.K_EQUALS:
                speed_multiplier += 0.1
            elif event.key == pygame.K_DOWN or event.key == pygame.K_MINUS:
                speed_multiplier = max(0.1, speed_multiplier - 0.1)
            elif event.key == pygame.K_r: # 按R重置
                 x = np.array([0, 0.2, 0, 0])
                 manual_force = 0.0
                 handle_rect.centerx = slider_rect.centerx
                 is_paused = False
            elif event.key == pygame.K_f:  # 按F切换是否应用手动施力
                apply_manual_force = not apply_manual_force
            elif event.key == pygame.K_l:  # 按L切换LQR是否激活
                is_lqr_active = not is_lqr_active

        # 鼠标事件 (滑块)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and handle_rect.collidepoint(event.pos):
                is_dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                is_dragging = False
        if event.type == pygame.MOUSEMOTION:
            if is_dragging:
                handle_rect.centerx = max(slider_rect.left, min(event.pos[0], slider_rect.right))
                # 将滑块位置映射到力的大小
                normalized_pos = (handle_rect.centerx - slider_rect.left) / slider_rect.width
                manual_force = (normalized_pos - 0.5) * 2 * slider_max_force

    # --- 2. 仿真逻辑更新 (如果未暂停) ---
    if not is_paused:
        # LQR控制器计算基础控制力
        if is_lqr_active:
            u_lqr = -K @ x
        else:
            u_lqr = np.array([0])  # LQR不激活时控制力为0

        # 计算总控制力
        if apply_manual_force:
            total_force = u_lqr[0] + manual_force
        else:
            total_force = u_lqr[0]

        # 对输入力矩进行限幅，防止力量过大
        force_clamped = np.clip(np.array([[total_force]]), -50.0, 50.0)

        # 使用状态空间方程计算状态的变化率
        x_dot = A @ x + (B @ force_clamped).flatten()

        # 用欧拉积分法更新状态，并考虑播放速度
        effective_dt = dt * speed_multiplier
        x = x + x_dot * effective_dt

    # --- 3. 绘图 (无论是否暂停都执行) ---
    screen.fill(C_WHITE)

    # 提取状态用于绘图
    cart_pos_m = x[0]
    pole_angle_rad = x[1]

    # --- 坐标转换 (从物理世界到屏幕) ---
    ground_y = HEIGHT - 100
    pygame.draw.line(screen, C_BLACK, (0, ground_y), (WIDTH, ground_y), 3)

    # 车体
    cart_x_px = WIDTH / 2 + cart_pos_m * SCALE
    cart_y_px = ground_y
    cart_w, cart_h = 100, 40
    rect_to_draw = pygame.Rect(int(cart_x_px - cart_w/2), int(cart_y_px - cart_h), cart_w, cart_h)
    pygame.draw.rect(screen, C_BLUE, rect_to_draw, border_radius=5)

    # 轮子
    wheel_radius_px = r_wheel * SCALE
    pygame.draw.circle(screen, C_BLACK, (int(cart_x_px - cart_w/2 * 0.7), int(cart_y_px - wheel_radius_px)), int(wheel_radius_px))
    pygame.draw.circle(screen, C_BLACK, (int(cart_x_px + cart_w/2 * 0.7), int(cart_y_px - wheel_radius_px)), int(wheel_radius_px))

    # 摆杆
    pole_len_px = L_pole * SCALE
    pole_base_x = cart_x_px
    pole_base_y = cart_y_px - cart_h
    pole_end_x = pole_base_x + pole_len_px * math.sin(pole_angle_rad)
    pole_end_y = pole_base_y - pole_len_px * math.cos(pole_angle_rad)
    pygame.draw.line(screen, C_RED, (int(pole_base_x), int(pole_base_y)), (int(pole_end_x), int(pole_end_y)), 8)
    pygame.draw.circle(screen, C_RED, (int(pole_end_x), int(pole_end_y)), 5)

    # --- 绘制滑块 ---
    pygame.draw.rect(screen, C_GREY, slider_rect, border_radius=10)
    pygame.draw.line(screen, C_DARK_GREY, (slider_rect.centerx, slider_rect.top), (slider_rect.centerx, slider_rect.bottom), 2)
    pygame.draw.rect(screen, C_BLUE, handle_rect, border_radius=5)

    # --- 显示信息 ---
    info_texts = [
        f"车体位置: {x[0]:.2f} m",
        f"摆杆角度: {math.degrees(x[1]):.2f} °",
        f"车体速度: {x[2]:.2f} m/s",
        f"摆杆角速度: {math.degrees(x[3]):.2f} °/s",
        f"LQR计算力: {(-K @ x)[0]:.2f} N" if is_lqr_active else "LQR计算力: 0.00 N (未激活)",
        f"手动施力: {manual_force:.2f} N",
        f"应用手动施力: {'是' if apply_manual_force else '否'} (按F切换)",
        f"LQR控制器: {'激活' if is_lqr_active else '关闭'} (按L切换)",
        f"仿真速度: {speed_multiplier:.1f}x (↑/↓)",
        f"按'R'键重置",
    ]
    for i, text in enumerate(info_texts):
        if "LQR控制器" in text:
            color = C_GREEN if is_lqr_active else C_RED
        elif "手动" in text:
            color = C_GREEN
        else:
            color = C_BLACK
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (15, 15 + i * 30))

    if is_paused:
        pause_font = pygame.font.SysFont("SimHei", 60)
        pause_surface = pause_font.render("已暂停", True, C_DARK_GREY)
        screen.blit(pause_surface, (WIDTH/2 - pause_surface.get_width()/2, HEIGHT/2 - 50))

    # 刷新屏幕
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()