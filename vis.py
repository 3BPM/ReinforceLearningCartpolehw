import matplotlib.pyplot as plt
import numpy as np
def plot_training_results(rewards, losses, lengths, window=100):
    plt.figure(figsize=(15, 5))

    # Reward plot
    plt.subplot(1, 3, 1)
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(rewards)), moving_avg, 'r-', label=f'{window}-ep Avg')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(losses, alpha=0.3, label='Training Loss')
    moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(losses)), moving_avg, 'r-', label=f'{window}-ep Avg')
    plt.title('Training Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()

    # Length plot
    plt.subplot(1, 3, 3)
    plt.plot(lengths, alpha=0.3, label='Episode Length')
    moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(lengths)), moving_avg, 'r-', label=f'{window}-ep Avg')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()

    plt.tight_layout()
    from datetime import datetime
    # 保存为当前时间文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plot_{current_time}.png"
    plt.savefig(filename)
    print(f"Saved as: {filename}")



def plot_system_response(system_response, time_vector):
    """绘制系统响应曲线
    
    参数:
    system_response -- 系统响应数据 (numpy数组)
    time_vector -- 时间向量 (numpy数组)
    """
    plt.figure(figsize=(10,8))
    titles = ['左轮转角', '右轮转角', '车身倾角', '摆杆倾角']
    for i in range(4):
        plt.subplot(4,1,i+1)
        plt.plot(time_vector, system_response[:,i], 'b', linewidth=1.5)
        plt.grid(True)
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()
    

def simulate_and_plot(initial_state, time_vector, closed_loop_matrix, input_matrix, input_signal, output_matrix):
    """模拟闭环系统并绘制响应
    
    参数:
    initial_state -- 初始状态向量 (numpy数组)
    time_vector -- 时间向量 (numpy数组)
    closed_loop_matrix -- 闭环系统状态矩阵 (numpy数组)
    input_matrix -- 输入矩阵 (numpy数组)
    input_signal -- 输入信号 (numpy数组)
    output_matrix -- 输出矩阵 (numpy数组)
    """
    state = initial_state
    system_response = np.zeros((len(time_vector), output_matrix.shape[0]))
    
    for i in range(len(time_vector)):
        system_response[i,:] = output_matrix @ state
        state = closed_loop_matrix @ state + input_matrix @ input_signal[:,i]
    
    plot_system_response(system_response, time_vector)