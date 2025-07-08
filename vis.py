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





def plot_system_outputs(system_responses, time_vectors=None, titles=None, n_subplots=None,
                       subplot_titles=['左轮转角', '右轮转角', '车身倾角', '摆杆倾角'],
                       line_styles=None, colors=None, figsize=(10, 8)):
    """
    绘制并比较多组系统响应曲线
    
    参数:
    system_responses -- 系统响应数据列表，每个元素是一个形状为(n,4)的numpy数组
    time_vectors -- 时间向量列表，每个元素是一个numpy数组。如果为None，则使用索引作为x轴
    titles -- 每组响应数据的图例标题列表
    subplot_titles -- 每个子图的标题列表
    line_styles -- 线型列表，如['-', '--', ':']
    colors -- 颜色列表，如['b', 'g', 'r']
    figsize -- 图形大小
    """
    if not isinstance(system_responses, list):
        system_responses = [system_responses]
    
    n_responses = len(system_responses)
    if n_subplots is None:
        n_subplots = system_responses[0].shape[1] if len(system_responses[0].shape) > 1 else 1
    
    # 设置默认值
    if time_vectors is None:
        time_vectors = [np.arange(len(resp)) for resp in system_responses]
    elif not isinstance(time_vectors, list):
        time_vectors = [time_vectors]
    
    if titles is None:
        titles = [f'响应 {i+1}' for i in range(n_responses)]
    
    if line_styles is None:
        line_styles = ['-'] * n_responses
    
    if colors is None:
        colors = np.random.rand(n_responses, 3)
    
    plt.figure(figsize=figsize)
    
    for i in range(n_subplots):
        plt.subplot(n_subplots, 1, i+1)
        
        for j in range(n_responses):
            # 处理单变量情况
            if len(system_responses[j].shape) == 1:
                y_data = system_responses[j]
            else:
                y_data = system_responses[j][:, i]
                
            plt.plot(time_vectors[j], y_data, 
                    linestyle=line_styles[j % len(line_styles)],
                    color=colors[j % len(colors)],
                    linewidth=1.5,
                    label=titles[j])
        
        plt.grid(True)
        if i < len(subplot_titles):
            plt.title(subplot_titles[i])
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_from_t_y_pairs(t_y_pairs_list,
                        titles=None,
                        subplot_titles=['左轮转角', '右轮转角', '车身倾角', '摆杆倾角'],
                        n_subplots=None,
                        line_styles=None,
                        colors=None,
                        figsize=(10, 8)):
    """
    在同一张图上绘制多组 (t, y) 数据对，支持每组数据有不同采样时间间隔。
    
    参数:
    t_y_pairs_list -- 列表，每个元素为 (t, y)，t 是时间向量，y 是 shape=(n,) 或 (n,4) 的数组
    titles -- 每组响应数据的图例名称
    subplot_titles -- 每个子图的标题（最多支持4个子图）
    line_styles -- 各曲线的线型列表，如 ['-', '--', ':']
    colors -- 各曲线的颜色列表，如 ['b', 'g', 'r']
    figsize -- 图形尺寸
    """
    if not isinstance(t_y_pairs_list, list):
        raise ValueError("t_y_pairs_list 必须是一个列表，元素为 (t, y) 对。")
    
    n_responses = len(t_y_pairs_list)
    
    # 推断子图数量
    sample_y = t_y_pairs_list[0][1]
    if n_subplots is None:
        n_subplots = sample_y.shape[1] if len(sample_y.shape) > 1 else 1
    
    if titles is None:
        titles = [f'响应 {i+1}' for i in range(n_responses)]
    
    if line_styles is None:
        line_styles = ['-'] * n_responses
    
    if colors is None:
        colors = [None] * n_responses  # 使用默认颜色
    
    plt.figure(figsize=figsize)
    
    for i in range(n_subplots):
        plt.subplot(n_subplots, 1, i+1)
        
        for j, (t, y) in enumerate(t_y_pairs_list):
            if len(y.shape) == 1 or y.ndim == 1:
                y_data = y
            else:
                y_data = y[:, i]
                
            plt.plot(t, y_data,
                     linestyle=line_styles[j % len(line_styles)],
                     color=colors[j % len(colors)] if colors[j % len(colors)] else None,
                     linewidth=1.5,
                     label=titles[j])
        
        plt.grid(True)
        if i < len(subplot_titles):
            plt.title(subplot_titles[i])
        plt.legend()
    
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
    
    plot_system_outputs(system_response, time_vector)

def print_a_matrix(A, ax=None, cmap='viridis', annotate=True, fmt=".2f", 
                   title="Matrix Visualization", xlabel="Columns", ylabel="Rows"):
    need_show=False
    if ax is None:
        ax = plt.gca()
        need_show=True
        
    im = ax.imshow(A, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    if annotate:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, format(A[i, j], fmt),
                        ha="center", va="center",
                        color="white" if A[i,j] < np.max(A)/2 else "black")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if need_show:
        plt.show()
def compare_matrices(K, data):
    if K.shape == data.shape or (K.T.shape == data.shape):
        K_comp = K.T if K.shape != data.shape else K
        diff = K_comp - data

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        print_a_matrix(K_comp, ax=axs[0], title="K_comp")
        print_a_matrix(data, ax=axs[1], title="data")
        print_a_matrix(diff, ax=axs[2], title="diff")

        plt.tight_layout()
        plt.show()

        print("K 与 data 的差异 (K - data):\n", diff)
        print("K 与 data 的最大绝对差值:", np.max(np.abs(diff)))
        print("K 与 data 的均方根误差(RMSE):", np.sqrt(np.mean(diff**2)))
        print("K 与 data 是否近似相等 (allclose):", np.allclose(K_comp, data))
    else:
        print("K 和 data 的形状不同: K.shape =", K.shape, ", data.shape =", data.shape)
