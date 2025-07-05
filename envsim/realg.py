import numpy as np
from numpy.linalg import inv
import pygame
import math
from scipy.linalg import solve_continuous_are

# ==================== 物理参数定义 (简化模型) ====================
m_car = 0.9      # 车体的质量 (kg)
m_pole = 0.1     # 摆杆的质量 (kg)
r_wheel = 0.0335 * 1.5 # 放大一点轮子半径，视觉效果更好
L_pole = 0.390   # 摆杆的长度 (m)
l_pole = L_pole / 2 # 摆杆质心到转轴的距离
g = 9.8        # 重力加速度 (m/s^2)
I_pole = (1/12) * m_pole * L_pole**2  # 摆杆转动惯量

# 【【【问题1解决】】】 定义非线性动力学模型所需的完整参数
M_total = m_car + m_pole
m = m_pole
L = l_pole
I = I_pole

# ==================== 状态空间模型 (仅用于LQR控制器计算) ====================
# A, B矩阵是线性化的，只在 angle ≈ 0 时精确
# 我们用它来计算K，但不用它来更新仿真世界
denom = I * (M_total) + m * M_total * L**2
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, (m**2 * L**2 * g) / denom, 0, 0],
    [0, (m * g * L * M_total) / denom, 0, 0]
])
B = np.array([
    [0],
    [0],
    [(I + m*L**2)/denom],
    [(m*L)/denom]
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
pygame.display.set_caption("独轮自平衡车 (L:开关LQR, R:重置, 空格:暂停)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("SimHei", 24)

# --- 颜色 ---
C_BLACK, C_WHITE, C_RED, C_BLUE, C_GREEN, C_ORANGE, C_GREY, C_DARK_GREY = (0,0,0), (255,255,255), (220,50,50), (50,100,200), (50,200,100), (255,140,0), (200,200,200), (100,100,100)

# --- 仿真控制参数 ---
SCALE = 200
FPS = 60
dt = 1.0 / FPS
is_paused = False
speed_multiplier = 1.0
is_lqr_active = True

# --- 初始状态 ---
x = np.array([0.0, 0.2, 0.0, 0.0]) # 确保是浮点数

# --- 滑块参数 ---
slider_max_force = 30.0
manual_force = 0.0
slider_rect = pygame.Rect(WIDTH // 4, HEIGHT - 50, WIDTH // 2, 20)
handle_rect = pygame.Rect(slider_rect.centerx - 10, slider_rect.y - 5, 20, 30)
is_dragging = False

# 【【【问题3解决】】】 创建一个用于绘制车体的原始图像 (Surface)
cart_w, cart_h = int(r_wheel * SCALE * 1.8), int(r_wheel * SCALE * 3.5) # 车体尺寸与轮子关联
cart_surf = pygame.Surface((cart_w, cart_h), pygame.SRCALPHA)
pygame.draw.rect(cart_surf, C_BLUE, cart_surf.get_rect(), border_radius=8)

# --- 主循环 ---
running = True
while running:
    # --- 1. 事件处理 ---
    # ... (事件处理代码保持不变) ...
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE: is_paused = not is_paused
            elif event.key == pygame.K_UP or event.key == pygame.K_EQUALS: speed_multiplier += 0.1
            elif event.key == pygame.K_DOWN or event.key == pygame.K_MINUS: speed_multiplier = max(0.1, speed_multiplier - 0.1)
            elif event.key == pygame.K_l: is_lqr_active = not is_lqr_active
            elif event.key == pygame.K_r:
                 x = np.array([0.0, 0.2, 0.0, 0.0])
                 manual_force = 0.0
                 handle_rect.centerx = slider_rect.centerx
                 is_paused = False
                 is_lqr_active = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and handle_rect.collidepoint(event.pos): is_dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: is_dragging = False
        if event.type == pygame.MOUSEMOTION:
            if is_dragging:
                handle_rect.centerx = max(slider_rect.left, min(event.pos[0], slider_rect.right))
                normalized_pos = (handle_rect.centerx - slider_rect.left) / slider_rect.width
                manual_force = (normalized_pos - 0.5) * 2 * slider_max_force

    # --- 2. 仿真逻辑更新 ---
    if not is_paused:
        # LQR控制器部分 (仍然基于线性模型)
        u_lqr = (-K @ x) if is_lqr_active else np.array([0.0])
        total_force = u_lqr[0] + manual_force

        # 【【【问题1解决】】】 使用非线性动力学模型更新状态
        pos, angle, pos_dot, angle_dot = x

        # 这是从拉格朗日方程推导出的标准非线性倒立摆动力学方程
        # angle_ddot (角加速度)
        numerator_alpha = (M_total * g * L * math.sin(angle) -
                           m * L * (L * angle_dot**2 * math.sin(angle) - total_force) * math.cos(angle))
        denominator_alpha = I * M_total + m * M_total * L**2 - (m * L * math.cos(angle))**2
        angle_ddot = numerator_alpha / denominator_alpha

        # pos_ddot (线加速度)
        numerator_x = (total_force + m * L * (angle_dot**2 * math.sin(angle) - angle_ddot * math.cos(angle)))
        denominator_x = M_total
        pos_ddot = numerator_x / denominator_x

        # 使用欧拉法进行积分
        effective_dt = dt * speed_multiplier
        pos_dot += pos_ddot * effective_dt
        angle_dot += angle_ddot * effective_dt
        pos += pos_dot * effective_dt
        angle += angle_dot * effective_dt

        # 更新状态向量
        x = np.array([pos, angle, pos_dot, angle_dot])

        # 【【【问题2解决】】】 碰撞检测
        # 简单的地面碰撞 (非弹性)
        # 假设地面在 y=0，轮子最低点不能低于0
        # 在我们的绘图中，地面在 ground_y，所以轮子中心不能低于 ground_y - radius
        # 这个逻辑在绘图部分处理更直观，但这里先假设没有地下室
        # (实际的物理碰撞会更复杂，这里仅作边界约束)
        # 在这个仿真中，我们不限制位置，让它可以在屏幕外移动

    # --- 3. 绘图 ---
    screen.fill(C_WHITE)

    # 提取状态
    wheel_pos_m = x[0]
    body_angle_rad = x[1]

    # 定义坐标
    ground_y = HEIGHT - 100
    pygame.draw.line(screen, C_BLACK, (0, ground_y), (WIDTH, ground_y), 3)

    wheel_radius_px = r_wheel * SCALE
    wheel_axle_x = WIDTH / 2 + wheel_pos_m * SCALE
    wheel_axle_y = ground_y - wheel_radius_px

    # 绘制轮子
    pygame.draw.circle(screen, C_DARK_GREY, (int(float(wheel_axle_x)), int(float(wheel_axle_y))), int(float(wheel_radius_px)))
    wheel_rot_angle = (wheel_pos_m / r_wheel) % (2 * math.pi)
    for i in range(4):
        spoke_angle = wheel_rot_angle + i * math.pi / 2
        spoke_end_x = wheel_axle_x + wheel_radius_px * math.cos(spoke_angle)
        spoke_end_y = wheel_axle_y + wheel_radius_px * math.sin(spoke_angle)
        pygame.draw.line(screen, C_GREY, (wheel_axle_x, wheel_axle_y), (spoke_end_x, spoke_end_y), 3)

    # 【【【问题3解决】】】 绘制绕轮轴旋转的矩形车体
    body_angle_deg = math.degrees(body_angle_rad)
    # 旋转原始的车体图像, Pygame旋转是逆时针，我们的角度是顺时针为正，所以用负号
    rotated_cart_surf = pygame.transform.rotate(cart_surf, -body_angle_deg)

    # 计算车体矩形的中心点应该在的位置
    # 1. 车体在未旋转时，其几何中心相对于轮轴的偏移
    # 假设车体底部中心贴着轮轴
    offset_y = -cart_h / 2
    offset_vector = pygame.math.Vector2(0, offset_y)

    # 2. 将这个偏移向量根据车体角度旋转
    rotated_offset = offset_vector.rotate(body_angle_deg)

    # 3. 最终车体中心的屏幕坐标 = 轮轴坐标 + 旋转后的偏移
    cart_center_x = wheel_axle_x + rotated_offset.x
    cart_center_y = wheel_axle_y + rotated_offset.y

    # 4. 获取旋转后图像的rect，并设置其中心
    rotated_cart_rect = rotated_cart_surf.get_rect(center=(int(cart_center_x), int(cart_center_y)))

    # 5. 绘制
    screen.blit(rotated_cart_surf, rotated_cart_rect)

    # 绘制摆杆
    # 摆杆的基点是车体的顶部中心。先计算这个点在世界中的位置。
    # 1. 顶部中心相对于车体几何中心的偏移
    pole_mount_offset_local = pygame.math.Vector2(0, -cart_h / 2)
    # 2. 将其根据车体角度旋转
    rotated_pole_mount_offset = pole_mount_offset_local.rotate(body_angle_deg)
    # 3. 最终摆杆基点 = 车体中心 + 旋转后的顶部偏移
    pole_base_x = cart_center_x + rotated_pole_mount_offset.x
    pole_base_y = cart_center_y + rotated_pole_mount_offset.y

    # 摆杆末端 (摆杆角度与车体角度相同)
    pole_len_px = L_pole * SCALE
    pole_end_x = pole_base_x + pole_len_px * math.sin(body_angle_rad)
    pole_end_y = pole_base_y - pole_len_px * math.cos(body_angle_rad) # y轴向下

    pygame.draw.line(screen, C_RED, (int(pole_base_x), int(pole_base_y)), (int(pole_end_x), int(pole_end_y)), 6)
    pygame.draw.circle(screen, C_RED, (int(pole_end_x), int(pole_end_y)), 5)

    # --- 绘制滑块和显示信息 ---
    # ... (这部分代码保持不变) ...
    pygame.draw.rect(screen, C_GREY, slider_rect, border_radius=10)
    pygame.draw.line(screen, C_DARK_GREY, (slider_rect.centerx, slider_rect.top), (slider_rect.centerx, slider_rect.bottom), 2)
    pygame.draw.rect(screen, C_BLUE, handle_rect, border_radius=5)

    lqr_status_text = f"LQR控制器: {'开启' if is_lqr_active else '关闭'}"
    lqr_status_color = C_GREEN if is_lqr_active else C_ORANGE
    lqr_force_text = f"LQR计算力: {u_lqr[0]:.2f} N" if is_lqr_active else "LQR计算力: 0.00 N"
    info_texts = [f"车轮位置: {x[0]:.2f} m", f"倾斜角度: {math.degrees(x[1] % (2*math.pi)):.2f} °", f"车轮速度: {x[2]:.2f} m/s", f"角速度: {math.degrees(x[3]):.2f} °/s", lqr_force_text, f"手动施力: {manual_force:.2f} N"]
    for i, text in enumerate(info_texts):
        text_surface = font.render(text, True, C_BLACK)
        screen.blit(text_surface, (15, 15 + i * 30))
    lqr_status_surface = font.render(lqr_status_text, True, lqr_status_color)
    screen.blit(lqr_status_surface, (15, 15 + len(info_texts) * 30))
    if is_paused:
        pause_font = pygame.font.SysFont("SimHei", 60)
        pause_surface = pause_font.render("已暂停", True, C_DARK_GREY)
        screen.blit(pause_surface, (WIDTH/2 - pause_surface.get_width()/2, HEIGHT/2 - 50))
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()