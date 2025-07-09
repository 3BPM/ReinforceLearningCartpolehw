import pygame
import math
import time # 用于演示
import numpy as np # 用于演示

from envsim.config import Config

# ==================== 渲染器类 (重构后) ====================
class UnicycleRenderer:
    def __init__(self, width=None, height=None):
        """初始化渲染器、Pygame、颜色和字体。"""

        self.width = width or Config.window_width
        self.height = height or Config.window_height
        self.scale = Config.scale
        self.ground_y = self.height - 100
        
        pygame.init()
        # 颜色定义
        self.colors = {
            'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (220, 50, 50),
            'blue': (50, 100, 200), 'green': (50, 200, 100), 'orange': (255, 140, 0),
            'grey': (200, 200, 200), 'dark_grey': (100, 100, 100),
            'purple': (128, 0, 128)
        }
        # 字体
        self.font = pygame.font.SysFont(Config.font_name, 24)
        self.small_font = pygame.font.SysFont(Config.font_name, 18)

    # ==================== 主要渲染方法 ====================

    def render(self, screen, simulator, is_paused=False):
        """
        【实时渲染】: 为实时仿真器渲染单个当前帧。
        从 simulator 对象获取当前状态并绘制。
        """
        state = simulator.get_state()
        # 实时渲染调用的是通用的帧渲染方法
        self._render_frame(screen, state) 
        # 然后绘制实时仿真特有的UI
        self._draw_simulation_ui(screen, simulator, is_paused)

    def playback_from_data(self, y_vector, time_vector, Ts=None):
        """
        【新增功能：回放渲染】: 根据提供的历史数据播放整个动画。
        
        Args:
            y_vector (list or np.ndarray): 状态向量的历史记录。
                                           每一行应包含 [theta_L, theta_R, theta_1, theta_2, ...]。
            time_vector (list or np.ndarray): 与 y_vector 对应的时间点。
            Ts (float): 采样时间，用于控制回放速度 (下一帧将在 Ts 秒后渲染)。
        """
        Ts=Ts or Config.dt
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Unicycle Simulation Playback")
        clock = pygame.time.Clock()
        
        running = True
        frame_index = 0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # 如果动画播放完毕，则停留在最后一帧
            if frame_index >= len(y_vector):
                frame_index = len(y_vector) - 1
            
            # 获取当前帧的状态和时间
            current_state = y_vector[frame_index]
            current_time = time_vector[frame_index]

            # 渲染当前帧
            self._render_frame(screen, current_state, playback_time=current_time)
            
            pygame.display.flip()

            # 移动到下一帧
            frame_index += 1
            
            # 控制回放速度，使得每帧的间隔为 Ts 秒
            # clock.tick()会根据1/Ts计算出FPS，并控制循环速度
            clock.tick(1 / Ts)
            
        pygame.quit()

    # ==================== 内部辅助方法 ====================
    
    def _render_frame(self, screen, state, playback_time=None):
        """
        【核心绘图逻辑】: 绘制单帧的物理实体（地面、轮子、车体、摆杆）。
        此方法被 render() 和 playback_from_data() 共享。
        
        Args:
            screen (pygame.Surface): 绘制的目标屏幕。
            state (list/tuple): 当前状态向量 [theta_L, theta_R, theta_1, theta_2, ...]。
            playback_time (float, optional): 如果是回放模式，传入当前时间以显示在UI上。
        """
        # 1. 清屏和绘制地面
        screen.fill(self.colors['white'])
        pygame.draw.line(screen, self.colors['black'], (0, self.ground_y), (self.width, self.ground_y), 3)

        # 2. 从状态向量中解包所需变量
        # 假设状态向量的顺序是: [theta_L, theta_R, theta_1, theta_2]
        theta_L = state[0]  # 轮子位置 (x)
        theta_1 = state[2]  # 车体/连杆1角度 (θ1)
        theta_2 = state[3]  # 摆杆/连杆2角度 (θ2)
        wheel_rot_angle = (theta_L / Config.r_wheel) % (2 * math.pi)

        # 3. 绘制各个组件
        self._draw_wheel(screen, theta_L, wheel_rot_angle)
        body_end_x, body_end_y = self._draw_body(screen, theta_L, theta_1)
        self._draw_pole(screen, body_end_x, body_end_y, theta_2)

        # 4. 如果是回放模式，绘制简化的UI
        if playback_time is not None:
            self._draw_playback_ui(screen, state, playback_time)

    def _draw_wheel(self, screen, wheel_pos_m, wheel_rot_angle):
        """绘制轮子"""
        wheel_center_x = self.width / 2 + wheel_pos_m * self.scale
        wheel_center_y = self.ground_y - Config.r_wheel * self.scale
        wheel_radius_px = Config.r_wheel * self.scale

        pygame.draw.circle(screen, self.colors['dark_grey'], (wheel_center_x, wheel_center_y), wheel_radius_px)
        for i in range(4):
            spoke_angle = wheel_rot_angle + i * math.pi / 2
            spoke_end_x = wheel_center_x + wheel_radius_px * math.cos(spoke_angle)
            spoke_end_y = wheel_center_y + wheel_radius_px * math.sin(spoke_angle)
            pygame.draw.line(screen, self.colors['grey'], (wheel_center_x, wheel_center_y), (spoke_end_x, spoke_end_y), 2)

    def _draw_body(self, screen, wheel_pos_m, body_angle_rad):
        """绘制车体并返回其末端坐标"""
        wheel_center_x = self.width / 2 + wheel_pos_m * self.scale
        wheel_center_y = self.ground_y - Config.r_wheel * self.scale
        body_len_px = Config.L_body * self.scale

        body_end_x = wheel_center_x + body_len_px * math.sin(body_angle_rad)
        body_end_y = wheel_center_y - body_len_px * math.cos(body_angle_rad) # Pygame Y轴向下

        pygame.draw.line(screen, self.colors['blue'], (wheel_center_x, wheel_center_y), (body_end_x, body_end_y), 8)
        pygame.draw.circle(screen, self.colors['blue'], (body_end_x, body_end_y), 6)
        return body_end_x, body_end_y

    def _draw_pole(self, screen, body_end_x, body_end_y, pole_angle_rad):
        """绘制摆杆"""
        pole_len_px = Config.L_pole * self.scale
        pole_end_x = body_end_x + pole_len_px * math.sin(pole_angle_rad)
        pole_end_y = body_end_y - pole_len_px * math.cos(pole_angle_rad)

        pygame.draw.line(screen, self.colors['red'], (body_end_x, body_end_y), (pole_end_x, pole_end_y), 6)
        pygame.draw.circle(screen, self.colors['red'], (pole_end_x, pole_end_y), 5)
        
    def _draw_playback_ui(self, screen, state, time):
        """为回放模式绘制简化的UI"""
        info_texts = [
            f"回放时间: {time:.2f} s",
            f"轮轴位置 (theta_L): {state[0]:.3f} m",
            f"车体角度 (theta_1): {math.degrees(state[2]):.2f} °",
            f"摆杆角度 (theta_2): {math.degrees(state[3]):.2f} °",
            "按 ESC 关闭窗口"
        ]
        for i, text in enumerate(info_texts):
            color = self.colors['dark_grey'] if "ESC" in text else self.colors['black']
            text_surface = self.font.render(text, True, color)
            screen.blit(text_surface, (15, 15 + i * 30))

    def _draw_simulation_ui(self, screen, simulator, is_paused):
        """为实时仿真模式绘制详细的UI（这是你原来的 _draw_ui 方法）"""
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
            f"车体角度: {math.degrees(state[2]):.2f} °",
            f"摆杆角度: {math.degrees(state[3]):.2f} °",
            f"仿真时间: {sim_time:.2f}s",
            # ... 其他信息 ...
        ]
        # (此处省略了完整的UI绘制代码，保持与你原始代码一致即可)
        lqr_status_text = f"LQR控制器: {'开启' if control_info['is_lqr_active'] else '关闭'}"
        lqr_force_text = f"LQR计算力: {control_info['lqr_force']:.2f} N"
        recording_text = f"数据记录: {'开启' if simulator.is_recording else '关闭'} (按D切换)"

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
