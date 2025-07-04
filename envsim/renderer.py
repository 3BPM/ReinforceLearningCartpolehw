import pygame
import math
from envsim.config import Config

# ==================== 渲染器类 ====================
class UnicycleRenderer:
    def __init__(self, width=None, height=None):
        self.width = width or Config.window_width
        self.height = height or Config.window_height
        self.scale = Config.scale
        self.ground_y = self.height - 100
        pygame.init()
        # 颜色定义
        self.colors = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red': (220, 50, 50),
            'blue': (50, 100, 200),
            'green': (50, 200, 100),
            'orange': (255, 140, 0),
            'grey': (200, 200, 200),
            'dark_grey': (100, 100, 100),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128)
        }

        # 字体
        self.font = pygame.font.SysFont(Config.font_name, 24)
        self.small_font = pygame.font.SysFont(Config.font_name, 18)

        # 创建车体Surface (这部分在当前逻辑中未使用，可以保留或移除)
        self.cart_w, self.cart_h = 40, 200
        self.cart_surf = pygame.Surface((self.cart_w, self.cart_h), pygame.SRCALPHA)
        pygame.draw.rect(self.cart_surf, self.colors['blue'], self.cart_surf.get_rect(), border_radius=8)
        pygame.draw.circle(self.cart_surf, self.colors['dark_grey'], (self.cart_w // 2, 5), 5)

    def render(self, screen, simulator, is_paused=False):
        """渲染整个场景"""
        # 清屏
        screen.fill(self.colors['white'])

        # 绘制地面
        pygame.draw.line(screen, self.colors['black'], (0, self.ground_y), (self.width, self.ground_y), 3)

        # 获取状态
        state = simulator.get_state()
        theta_L = state[0]      # 轮子位置 (x)
        # theta_R = state[1]    # (未使用)
        theta_1 = state[2]      # 连杆1角度/车体角度 (θ1)
        theta_2 = state[3]      # 连杆2角度/摆杆角度 (θ2)
        wheel_rot_angle = (theta_L / Config.r_wheel) % (2 * math.pi)

        # 绘制轮子
        self._draw_wheel(screen, theta_L, wheel_rot_angle)

        # 【修改点 1】: 绘制车体，并捕获其末端坐标
        body_end_x, body_end_y = self._draw_body(screen, theta_L, theta_1)

        # 【修改点 2】: 使用正确的坐标来绘制摆杆
        self._draw_pole(screen, body_end_x, body_end_y, theta_2)

        # 绘制UI
        self._draw_ui(screen, simulator, is_paused)

    def render_cartpole(self, screen, state, theta_L, theta_R, theta_1, theta_2):
        """渲染小车倒立摆系统"""
        # 清屏
        screen.fill(self.colors['white'])

        # 绘制地面
        pygame.draw.line(screen, self.colors['black'], (0, self.ground_y), (self.width, self.ground_y), 3)

        # 获取状态
        theta_L = theta_L      # 轮子位置 (x)
        # theta_R = state[1]    # (未使用)

        wheel_rot_angle = (theta_L / Config.r_wheel) % (2 * math.pi)

        # 绘制轮子
        self._draw_wheel(screen, theta_L, wheel_rot_angle)

        # 【修改点 1】: 绘制车体，并捕获其末端坐标
        body_end_x, body_end_y = self._draw_body(screen, theta_L, theta_1)

        # 【修改点 2】: 使用正确的坐标来绘制摆杆
        self._draw_pole(screen, body_end_x, body_end_y, theta_2)

    def _draw_wheel(self, screen, wheel_pos_m, wheel_rot_angle):
        """绘制轮子"""
        wheel_center_x = self.width / 2 + wheel_pos_m * self.scale
        wheel_center_y = self.ground_y - Config.r_wheel * self.scale
        wheel_radius_px = Config.r_wheel * self.scale

        # 画轮子
        pygame.draw.circle(screen, self.colors['dark_grey'],
                          (int(wheel_center_x), int(wheel_center_y)), int(wheel_radius_px))

        # 画轮子的十字和旋转线
        for i in range(4):
            spoke_angle = wheel_rot_angle + i * math.pi / 2
            spoke_end_x = wheel_center_x + wheel_radius_px * math.cos(spoke_angle)
            spoke_end_y = wheel_center_y + wheel_radius_px * math.sin(spoke_angle)
            pygame.draw.line(screen, self.colors['grey'],
                           (wheel_center_x, wheel_center_y), (spoke_end_x, spoke_end_y), 2)

    def _draw_body(self, screen, wheel_pos_m, body_angle_rad):
        """绘制车体"""
        wheel_center_x = self.width / 2 + wheel_pos_m * self.scale
        wheel_center_y = self.ground_y - Config.r_wheel * self.scale

        # 车体修长的杆
        body_len_px = 0.126 * self.scale  # 注意: 这个长度最好也放入Config中
        body_start_x = wheel_center_x
        body_start_y = wheel_center_y
        body_end_x = wheel_center_x + body_len_px * math.sin(body_angle_rad)
        body_end_y = wheel_center_y - body_len_px * math.cos(body_angle_rad) # Pygame Y轴向下，-cos是向上

        # 绘制车体杆
        pygame.draw.line(screen, self.colors['blue'],
                        (body_start_x, body_start_y), (body_end_x, body_end_y), 8)
        pygame.draw.circle(screen, self.colors['blue'],
                          (int(body_end_x), int(body_end_y)), 6)

        # 【修改点 3】: 返回车体末端坐标，供摆杆使用
        return body_end_x, body_end_y

    def _draw_pole(self, screen, body_end_x, body_end_y, pole_angle_rad):
        """绘制摆杆，基于车体末端的位置和角度"""
        # 摆杆
        pole_len_px = Config.L_pole * self.scale
        # 摆杆的起点现在是正确的车体末端坐标
        pole_start_x = body_end_x
        pole_start_y = body_end_y
        # 基于起点和摆杆自身角度计算终点
        pole_end_x = pole_start_x + pole_len_px * math.sin(pole_angle_rad)
        pole_end_y = pole_start_y - pole_len_px * math.cos(pole_angle_rad)

        pygame.draw.line(screen, self.colors['red'],
                        (pole_start_x, pole_start_y), (pole_end_x, pole_end_y), 6)
        pygame.draw.circle(screen, self.colors['red'],
                          (int(pole_end_x), int(pole_end_y)), 5)

    def _draw_ui(self, screen, simulator, is_paused):
        """绘制UI界面"""
        # 注意：这里的state索引可能与render函数中的不一致，请根据你的模拟器实际状态向量进行调整
        state = simulator.get_state()
        control_info = simulator.get_control_info()
        sim_time = simulator.get_simulation_time()

        # 信息文本
        info_texts = [
            f"轮轴位置: {state[0]:.2f} m",        # x
            f"车体角度: {math.degrees(state[2]):.2f} °", # θ1
            f"摆杆角度: {math.degrees(state[3]):.2f} °", # θ2
            # 下面是可能的导数项，如果你的state向量包含它们
            # f"轮轴速度: {state[...]:.2f} m/s",
            # f"车体角速度: {math.degrees(state[...]):.2f} °/s",
            f"手动施力: {control_info['manual_force']:.2f} N",
            f"应用手动施力: {'是' if control_info['apply_manual_force'] else '否'} (按F切换)",
            f"仿真速度: {simulator.speed_multiplier:.2f}x",
            f"仿真时间: {sim_time:.2f}s"
        ]

        # 为了清晰，我调整了UI部分的文本，使其与render函数中的状态变量对应
        # 原代码中的UI部分可能存在state索引错误
        original_ui_texts = [
            f"轮轴位置: {state[0]:.2f} m",
            f"整体角度: {math.degrees(state[1]):.2f} °", # 这里可能是个错误，因为render中未使用state[1]
            f"轮轴速度: {state[2]:.2f} m/s",      # 这里与render中state[2]是角度的定义冲突
            f"角速度: {math.degrees(state[3]):.2f} °/s", # 这里与render中state[3]是角度的定义冲突
        ]


        lqr_status_text = f"LQR控制器: {'开启' if control_info['is_lqr_active'] else '关闭'}"
        lqr_force_text = f"LQR计算力: {control_info['lqr_force']:.2f} N"
        recording_text = f"数据记录: {'开启' if simulator.is_recording else '关闭'} (按D切换)"

        # 绘制信息文本 (使用调整后的版本)
        for i, text in enumerate(info_texts):
            color = self.colors['green'] if "是" in text else self.colors['black']
            text_surface = self.font.render(text, True, color)
            screen.blit(text_surface, (15, 15 + i * 30))

        lqr_status_surface = self.font.render(lqr_status_text, True,
                                            self.colors['green'] if control_info['is_lqr_active'] else self.colors['orange'])
        screen.blit(lqr_status_surface, (15, 15 + len(info_texts) * 30))

        lqr_force_surface = self.font.render(lqr_force_text, True,
                                           self.colors['blue'] if control_info['is_lqr_active'] else self.colors['grey'])
        screen.blit(lqr_force_surface, (15, 15 + (len(info_texts)+1) * 30))

        recording_color = self.colors['purple'] if simulator.is_recording else self.colors['grey']
        recording_surface = self.small_font.render(recording_text, True, recording_color)
        screen.blit(recording_surface, (15, 15 + (len(info_texts)+2) * 30))

        if is_paused:
            pause_surface = self.font.render("已暂停", True, self.colors['dark_grey'])
            screen.blit(pause_surface, (self.width / 2 - pause_surface.get_width() / 2, self.height / 2 - 50))

        help_texts = [
            "操作说明:", "L: 切换LQR控制器", "F: 切换手动施力", "R: 重置仿真", "空格: 暂停/继续",
            "↑/↓: 调整仿真速度", "D: 开始/停止数据记录", "A: 生成分析报告"
        ]
        for i, text in enumerate(help_texts):
            color = self.colors['dark_grey'] if i == 0 else self.colors['grey']
            font = self.font if i == 0 else self.small_font
            text_surface = font.render(text, True, color)
            screen.blit(text_surface, (self.width - 200, 15 + i * 25))