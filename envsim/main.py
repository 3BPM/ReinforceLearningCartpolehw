import pygame
from lqr_controller import LQRController
from simulator import UnicycleSimulator
from renderer import UnicycleRenderer
from input_handler import InputHandler

# ==================== 主程序 ====================
def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 700))
    pygame.display.set_caption("独轮自平衡车 (按L切换开关LQR,按F切换是否应用手动施力, R:重置,  空格:暂停, ↑/↓:速度)")
    clock = pygame.time.Clock()
    
    # 创建各个组件
    controller = LQRController()
    simulator = UnicycleSimulator(controller)
    renderer = UnicycleRenderer()
    input_handler = InputHandler(simulator, renderer)
    
    # 主循环
    running = True
    
    while running:
        # 处理输入
        running = input_handler.handle_events()
        
        # 更新仿真
        if not input_handler.is_paused:
            simulator.step()
        
        # 渲染
        renderer.render(screen, simulator, input_handler.is_paused)
        input_handler.draw_slider(screen)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main() 