import pygame
from envsim.lqr_controller import LQRController
from envsim.simulator import UnicycleSimulator
from envsim.renderer import UnicycleRenderer
from envsim.input_handler import InputHandler
from envsim.result_analyzer import ResultAnalyzer
from envsim.config import Config

# ==================== 主程序 ====================
def main():
    pygame.init()
    screen = pygame.display.set_mode((Config.window_width, Config.window_height))
    pygame.display.set_caption("独轮自平衡车LQR控制仿真 (按D开始记录数据,按A生成分析报告)")
    clock = pygame.time.Clock()

    # 创建结果分析器
    result_analyzer = ResultAnalyzer()

    # 创建各个组件
    controller = LQRController()
    simulator = UnicycleSimulator(controller, result_analyzer)
    renderer = UnicycleRenderer()
    input_handler = InputHandler(simulator, renderer)

    print("=== 独轮自平衡车LQR控制仿真 ===")
    print("操作说明:")
    print("- L: 切换LQR控制器")
    print("- F: 切换手动施力")
    print("- R: 重置仿真")
    print("- 空格: 暂停/继续")
    print("- ↑/↓: 调整仿真速度")
    print("- D: 开始/停止数据记录")
    print("- A: 生成分析报告")
    print("================================")

    # 主循环
    running = True

    while running:
        # 处理输入
        running = input_handler.handle_events()

        # 更新仿真
        if not input_handler.is_paused:
            input_handler.is_paused=simulator.step()

        # 渲染
        renderer.render(screen, simulator, input_handler.is_paused)
        input_handler.draw_slider(screen)

        pygame.display.flip()
        clock.tick(Config.fps)

    pygame.quit()

if __name__ == "__main__":
    main()