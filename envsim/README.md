# 独轮自平衡车仿真 - 模块化结构

## 文件结构

```
envsim/
├── __init__.py          # 包初始化文件
├── config.py            # 配置参数
├── lqr_controller.py    # LQR控制器
├── simulator.py         # 仿真器
├── renderer.py          # 渲染器
├── input_handler.py     # 输入处理器
├── result_analyzer.py   # 结果分析器
├── main.py             # 主程序
└── README.md           # 说明文档
```

## 模块说明

### 1. Config (config.py)
- **职责**: 集中管理所有配置参数
- **包含**: 物理参数、仿真参数、控制参数、可视化参数
- **特点**: 静态类，便于全局访问和修改

### 2. LQRController (lqr_controller.py)
- **职责**: LQR控制器的设计和计算
- **包含**: 系统矩阵A、B，控制增益K，控制力计算
- **方法**: 
  - `compute_control(state)`: 计算LQR控制力
  - `get_system_matrices()`: 获取系统矩阵

### 3. UnicycleSimulator (simulator.py)
- **职责**: 仿真状态管理和动力学计算
- **包含**: 状态更新、控制逻辑、参数设置、数据记录
- **方法**:
  - `step()`: 执行一步仿真
  - `reset()`: 重置仿真状态
  - `start_recording()`/`stop_recording()`: 数据记录控制
  - `set_*()`: 各种参数设置方法

### 4. UnicycleRenderer (renderer.py)
- **职责**: 所有Pygame绘制逻辑
- **包含**: 轮子、车体、摆杆绘制，UI显示，操作提示
- **方法**:
  - `render()`: 主渲染方法
  - `_draw_wheel()`, `_draw_body()`, `_draw_pole()`: 各组件绘制
  - `_draw_ui()`: UI界面绘制

### 5. InputHandler (input_handler.py)
- **职责**: 用户输入处理
- **包含**: 键盘事件、鼠标事件、滑块交互、分析功能
- **方法**:
  - `handle_events()`: 处理所有输入事件
  - `_handle_keyboard()`, `_handle_mouse()`: 分类处理
  - `draw_slider()`: 绘制滑块

### 6. ResultAnalyzer (result_analyzer.py) ⭐ 新功能
- **职责**: 仿真结果分析和可视化
- **包含**: 性能指标计算、响应曲线绘制、相图分析、报告生成
- **方法**:
  - `add_data_point()`: 添加数据点
  - `calculate_performance_metrics()`: 计算性能指标
  - `plot_detailed_response()`: 绘制详细响应曲线
  - `plot_phase_portrait()`: 绘制相图
  - `generate_report()`: 生成分析报告

## 使用方法

### 方法1: 实时仿真
```bash
cd envsim
python main.py
```

### 方法2: 离线分析
```bash
python example_analysis.py
```

### 方法3: 模块化使用
```python
from envsim import LQRController, UnicycleSimulator, UnicycleRenderer, InputHandler, ResultAnalyzer

# 创建组件
controller = LQRController()
analyzer = ResultAnalyzer()
simulator = UnicycleSimulator(controller, analyzer)
renderer = UnicycleRenderer()
input_handler = InputHandler(simulator, renderer)

# 使用各个组件...
```

## 操作说明

### 基本操作
- **L**: 切换LQR控制器开启/关闭
- **F**: 切换是否应用手动施力
- **R**: 重置仿真状态
- **空格**: 暂停/继续仿真
- **↑/↓**: 调整仿真速度

### 分析功能 ⭐ 新功能
- **D**: 开始/停止数据记录
- **A**: 生成分析报告（需要先记录数据）

### 鼠标操作
- 拖动滑块：调整手动施力大小

## 分析功能详解

### 性能指标
- **最大偏差**: 各状态变量的最大偏差
- **稳态误差**: 仿真后期的平均误差
- **调节时间**: 系统达到稳态所需时间
- **超调量**: 响应的最大超调百分比
- **控制性能**: 控制力大小和能量消耗

### 可视化输出
1. **详细响应曲线**: 包含所有状态变量和控制力的时间响应
2. **相图分析**: 状态变量之间的相图关系
3. **性能指标表格**: 关键性能参数的汇总

### 生成文件
- `simulation_report.txt`: 详细的分析报告
- `response_analysis.png`: 响应曲线图
- `phase_portrait.png`: 相图分析

## 优势

1. **模块化**: 每个类都有明确的职责
2. **可维护性**: 修改某个功能只需要修改对应的模块
3. **可扩展性**: 可以轻松添加新的控制器或渲染方式
4. **可测试性**: 每个组件都可以独立测试
5. **可重用性**: 控制器和仿真器可以用于其他项目
6. **分析能力**: 内置强大的结果分析功能 ⭐

## 依赖

- numpy
- pygame
- scipy
- matplotlib

安装依赖：
```bash
pip install numpy pygame scipy matplotlib
```

## 示例输出

运行 `example_analysis.py` 将生成：
- 详细的性能指标计算
- 专业的响应曲线图
- 相图分析
- 完整的分析报告

这些功能使得仿真不仅能够实时运行，还能进行深入的性能分析和评估。 