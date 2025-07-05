import pygame

# ==================== 输入处理器类 ====================
class InputHandler:
    def __init__(self, simulator, renderer):
        self.simulator = simulator
        self.renderer = renderer
        self.is_dragging = False
        self.is_paused = True
        
        # 滑块参数
        self.slider_max_force = 30.0
        self.slider_rect = pygame.Rect(renderer.width // 4, renderer.height - 50, 
                                     renderer.width // 2, 20)
        self.handle_rect = pygame.Rect(self.slider_rect.centerx - 10, 
                                      self.slider_rect.y - 5, 20, 30)

    def handle_events(self):
        """处理所有输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self._handle_keyboard(event)
            self._handle_mouse(event)
        return True

    def _handle_keyboard(self, event):
        """处理键盘事件"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == pygame.K_EQUALS:
                self.simulator.set_speed_multiplier(self.simulator.speed_multiplier + 0.01)
            elif event.key == pygame.K_DOWN or event.key == pygame.K_MINUS:
                self.simulator.set_speed_multiplier(self.simulator.speed_multiplier - 0.01)
            elif event.key == pygame.K_SPACE:
                self.is_paused = not self.is_paused
            elif event.key == pygame.K_l:
                self.simulator.set_lqr_active(not self.simulator.is_lqr_active)
            elif event.key == pygame.K_f:
                self.simulator.set_apply_manual_force(not self.simulator.apply_manual_force)
            elif event.key == pygame.K_r:
                self.simulator.reset()
                self.handle_rect.centerx = self.slider_rect.centerx

    def _handle_mouse(self, event):
        """处理鼠标事件"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.handle_rect.collidepoint(event.pos):
                self.is_dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.is_dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.is_dragging:
                self.handle_rect.centerx = max(self.slider_rect.left, 
                                             min(event.pos[0], self.slider_rect.right))
                normalized_pos = (self.handle_rect.centerx - self.slider_rect.left) / self.slider_rect.width
                manual_force = (normalized_pos - 0.5) * 2 * self.slider_max_force
                self.simulator.set_manual_force(manual_force)

    def draw_slider(self, screen):
        """绘制滑块"""
        pygame.draw.rect(screen, self.renderer.colors['grey'], self.slider_rect, border_radius=10)
        pygame.draw.line(screen, self.renderer.colors['dark_grey'], 
                        (self.slider_rect.centerx, self.slider_rect.top), 
                        (self.slider_rect.centerx, self.slider_rect.bottom), 2)
        pygame.draw.rect(screen, self.renderer.colors['blue'], self.handle_rect, border_radius=5) 