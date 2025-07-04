import numpy as np
from numpy.linalg import inv
import pygame
import math
from scipy.linalg import solve_continuous_are

# ==================== 物理参数定义 ====================
m_car = 0.9       # 车体质量
m_pole = 0.1      # 摆杆质量
r_wheel = 0.0335 * 1.5  # 视觉放大后的轮子半径
L_pole = 0.390    # 摆杆长度
l_pole = L_pole / 2
g = 9.8
I_pole = (1/12) * m_pole * L_pole**2  # 摆杆惯量

# ==================== 状态空间模型 ====================
M = m_car + m_pole
m = m_pole
L = l_pole
I = I_pole
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
    [(I + m * L**2) / denom],
    [(m * L) / denom]
])

# ==================== LQR控制器设计 ====================
Q = np.diag([1.0, 100.0, 1.0, 1.0])
R = np.array([[0.1]])
P = solve_continuous_are(A, B, Q, R)
K = inv(R) @ B.T @ P
print("计算得到的LQR增益矩阵 K:", K)

# ==================== Pygame仿真与可视化 ====================
pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("独轮自平衡车 (L:开关LQR, R:重置, 空格:暂停, ↑/↓:速度)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("SimHei", 24)

# --- 颜色 ---
C_BLACK = (0, 0, 0)
C_WHITE = (255, 255, 255)
C_RED = (220, 50, 50)
C_BLUE = (50, 100, 200)
C_GREEN = (50, 200, 100)
C_ORANGE = (255, 140, 0)
C_GREY = (200, 200, 200)
C_DARK_GREY = (100, 100, 100)

# --- 参数 ---
SCALE = 250           # 坐标缩放比例
FPS = 60
dt = 1.0 / FPS
is_paused = True
speed_multiplier = 1.0
is_lqr_active = True

# --- 初始状态 ---
x = np.array([0, 0.2, 0, 0])  # 初始位置、角度、速度、角速度

# --- 滑块参数 ---
slider_max_force = 30.0
manual_force = 0.0
slider_rect = pygame.Rect(WIDTH // 4, HEIGHT - 50, WIDTH // 2, 20)
handle_rect = pygame.Rect(slider_rect.centerx - 10, slider_rect.y - 5, 20, 30)
is_dragging = False

# --- 创建车体Surface ---
cart_w, cart_h = 40, 200
cart_surf = pygame.Surface((cart_w, cart_h), pygame.SRCALPHA)
pygame.draw.rect(cart_surf, C_BLUE, cart_surf.get_rect(), border_radius=8)
pygame.draw.circle(cart_surf, C_DARK_GREY, (cart_w // 2, 5), 5)

# --- 主循环 ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_paused = not is_paused
            elif event.key == pygame.K_UP or event.key == pygame.K_EQUALS:
                speed_multiplier += 0.1
            elif event.key == pygame.K_DOWN or event.key == pygame.K_MINUS:
                speed_multiplier = max(0.1, speed_multiplier - 0.1)
            elif event.key == pygame.K_l:
                is_lqr_active = not is_lqr_active
            elif event.key == pygame.K_r:
                x = np.array([0, 0.2, 0, 0])
                manual_force = 0.0
                handle_rect.centerx = slider_rect.centerx
                is_paused = False
                is_lqr_active = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and handle_rect.collidepoint(event.pos):
                is_dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                is_dragging = False
        if event.type == pygame.MOUSEMOTION:
            if is_dragging:
                handle_rect.centerx = max(slider_rect.left, min(event.pos[0], slider_rect.right))
                normalized_pos = (handle_rect.centerx - slider_rect.left) / slider_rect.width
                manual_force = (normalized_pos - 0.5) * 2 * slider_max_force

    # 更新状态
    if not is_paused:
        u_lqr = (-K @ x) if is_lqr_active else np.array([0.0])
        total_force = u_lqr[0] + manual_force
        force_clamped = np.clip(np.array([[total_force]]), -50.0, 50.0)
        x_dot = A @ x + (B @ force_clamped).flatten()
        effective_dt = dt * speed_multiplier
        x = x + x_dot * effective_dt

    screen.fill(C_WHITE)

    # 地面和坐标参考
    ground_y = HEIGHT - 100
    pygame.draw.line(screen, C_BLACK, (0, ground_y), (WIDTH, ground_y), 3)

    # 轮子物理参数
    wheel_pos_m = x[0]
    body_angle_rad = x[1]
    wheel_rot_angle = (wheel_pos_m / r_wheel) % (2 * math.pi)

    # 轮子中心
    wheel_center_x = WIDTH / 2 + wheel_pos_m * SCALE
    wheel_center_y = ground_y - r_wheel * SCALE
    wheel_radius_px = r_wheel * SCALE

    # 画轮子
    pygame.draw.circle(screen, C_DARK_GREY, (int(wheel_center_x), int(wheel_center_y)), int(wheel_radius_px))
    # 画轮子的十字和旋转线
    for i in range(4):
        spoke_angle = wheel_rot_angle + i * math.pi / 2
        spoke_end_x = wheel_center_x + wheel_radius_px * math.cos(spoke_angle)
        spoke_end_y = wheel_center_y + wheel_radius_px * math.sin(spoke_angle)
        pygame.draw.line(screen, C_GREY, (wheel_center_x, wheel_center_y), (spoke_end_x, spoke_end_y), 2)

    # 车体修长的杆
    body_len_px = 0.126 * SCALE
    body_start_x = wheel_center_x
    body_start_y = wheel_center_y
    body_end_x = wheel_center_x + body_len_px * math.sin(body_angle_rad)
    body_end_y = wheel_center_y - body_len_px * math.cos(body_angle_rad)

    # 绘制车体杆
    pygame.draw.line(screen, C_BLUE, (body_start_x, body_start_y), (body_end_x, body_end_y), 8)
    pygame.draw.circle(screen, C_BLUE, (int(body_end_x), int(body_end_y)), 6)

    # 摆杆
    pole_len_px = L_pole * SCALE
    pole_start_x = body_end_x
    pole_start_y = body_end_y
    pole_end_x = pole_start_x + pole_len_px * math.sin(body_angle_rad)
    pole_end_y = pole_start_y - pole_len_px * math.cos(body_angle_rad)

    pygame.draw.line(screen, C_RED, (pole_start_x, pole_start_y), (pole_end_x, pole_end_y), 6)
    pygame.draw.circle(screen, C_RED, (int(pole_end_x), int(pole_end_y)), 5)

    # 滑块接口
    pygame.draw.rect(screen, C_GREY, slider_rect, border_radius=10)
    pygame.draw.line(screen, C_DARK_GREY, (slider_rect.centerx, slider_rect.top), (slider_rect.centerx, slider_rect.bottom), 2)
    pygame.draw.rect(screen, C_BLUE, handle_rect, border_radius=5)

    # 显示信息
    info_texts = [
        f"轮轴位置: {wheel_pos_m:.2f} m",
        f"整体角度: {math.degrees(body_angle_rad):.2f} °",
        f"轮轴速度: {x[2]:.2f} m/s",
        f"角速度: {math.degrees(x[3]):.2f} °/s",
        f"手动施力: {manual_force:.2f} N",
        f"仿真速度: {speed_multiplier:.1f}x"
    ]

    lqr_status_text = f"LQR控制器: {'开启' if is_lqr_active else '关闭'}"
    lqr_force_text = f"LQR计算力: {(-K @ x)[0]:.2f} N" if is_lqr_active else "LQR计算力: 0.00 N"

    for i, text in enumerate(info_texts):
        color = C_GREEN if "手动" in text else C_BLACK
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (15, 15 + i * 30))

    lqr_status_surface = font.render(lqr_status_text, True, C_GREEN if is_lqr_active else C_ORANGE)
    screen.blit(lqr_status_surface, (15, 15 + len(info_texts) * 30))
    lqr_force_surface = font.render(lqr_force_text, True, C_BLUE if is_lqr_active else C_GREY)
    screen.blit(lqr_force_surface, (15, 15 + (len(info_texts)+1) * 30))

    if is_paused:
        pause_font = pygame.font.SysFont("SimHei", 60)
        pause_surface = pause_font.render("已暂停", True, C_DARK_GREY)
        screen.blit(pause_surface, (WIDTH / 2 - pause_surface.get_width() / 2, HEIGHT / 2 - 50))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
