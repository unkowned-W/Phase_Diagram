import matplotlib.pyplot as plt
import numpy as np

def create_prediction_grid(fixed_dims, fixed_vals, resolution=300):
    """创建四维预测网格，其中两个维度固定为具体值，两个维度在[0,1]上均匀布点"""
    if len(fixed_dims) != 2 or len(fixed_vals) != 2:
        raise ValueError("fixed_dims 和 fixed_vals 必须都是长度为2的列表")
    
    # 生成自由维度的网格点
    free_dim_vals = np.linspace(0, 1, resolution)
    
    # 创建所有自由维度的组合
    free_grid = np.array([[xi, yi] for xi in free_dim_vals for yi in free_dim_vals])
    
    # 创建四维网格点
    n_points = free_grid.shape[0]  # resolution^2
    X_grid = np.zeros((n_points, 4))
    
    # 设置固定维度
    X_grid[:, fixed_dims[0]] = fixed_vals[0]
    X_grid[:, fixed_dims[1]] = fixed_vals[1]
    
    # 找到自由维度的索引
    free_dims = [i for i in range(4) if i not in fixed_dims]
    
    # 设置自由维度
    X_grid[:, free_dims[0]] = free_grid[:, 0]
    X_grid[:, free_dims[1]] = free_grid[:, 1]
    
    grid_shape = (resolution, resolution)
    
    return X_grid, grid_shape

def plot_performance(all_performance):
    mean_perf = all_performance.mean(axis=0)
    std_perf = all_performance.std(axis=0)

        # Optional: metric names (update as needed)
    metric_names = ["Accuracy", "Balanced Accuracy", "F1 Score", "AUC"]

        # Plot with error bars
    plt.figure(figsize=(8, 8))
    plt.bar(range(len(mean_perf)), mean_perf, yerr=std_perf, color = "blue", capsize=10, linewidth=1, edgecolor='black')
    plt.xticks(ticks=np.arange(len(mean_perf)), labels=metric_names, rotation=45, fontsize = 18)
    plt.ylabel("Performance", fontsize = 18)
    plt.yticks(fontsize = 12)
    plt.tight_layout()
    plt.savefig("./tree_performance.png", dpi = 300)
    plt.show()

    print(mean_perf)
    print(std_perf)
    return mean_perf, std_perf

def plot_feature_importance(feature_importance, feature_names):
    mean_importance = feature_importance.mean(axis=0)
    std_importance = feature_importance.std(axis=0)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.bar(range(len(mean_importance)), mean_importance, yerr=std_importance, color = "blue", capsize=10, linewidth=1, edgecolor='black')
    plt.xticks(ticks=range(len(mean_importance)), labels=feature_names, rotation=45, fontsize = 18)
    plt.ylabel("Feature Importance", fontsize = 18)
    plt.yticks(fontsize = 12)
    plt.tight_layout()
    plt.savefig("./feature_importance_from_tree.png", dpi = 300)
    plt.show()

    print(mean_importance)
    print(std_importance)
    return mean_importance, std_importance

def plot_phase_diagram(model, X_grid, grid_shape, X, Y, dimension_index, feature_names):
    """使用预创建的网格绘制相图"""
    # 使用模型预测概率
    score = model.predict_proba(X_grid)
    
    # 确定用于绘图的两个维度
    dim1, dim2 = dimension_index
    
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_axes([0.12, 0.09, 0.7, 0.7])
    
    # 重新形状化为网格形状（注意转置以正确显示）
    probability_grid = score[:, 1].reshape(grid_shape).T
    
    # 计算显示范围（假设自由维度在[0,1]范围内）
    extent = [0, 1, 0, 1]
    
    im = ax.imshow(
        probability_grid,
        cmap='bwr',
        origin='lower',
        aspect='auto',
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        zorder=5
    )
    
    cb = plt.colorbar(im)
    cb.set_label('P (crystallization)', fontsize=18)
    ax.set_xlabel(feature_names[dim1], fontsize=18)
    ax.set_ylabel(feature_names[dim2], fontsize=18)
    
    # 绘制原始数据点（注意：X应只包含用于绘图的两个维度）
    ax.scatter(
        X[Y == True].T[0], 
        X[Y == True].T[1], 
        edgecolors='black', 
        linewidths=0.5, 
        c='red', 
        zorder=10
    )
    ax.scatter(
        X[Y == False].T[0], 
        X[Y == False].T[1], 
        edgecolors='black', 
        linewidths=0.5, 
        c='blue', 
        zorder=10
    )
    
    return fig, ax

def plot_batch_performance(
    mean_value,  # 核心输入：批次×指标的均值（二维数组/列表）
    std_value,   # 核心输入：批次×指标的标准差（同结构）
    title="Model Performance by Batch",  # 自定义标题
    metrics=["Accuracy", "Balanced Accuracy", "F1 Score", "AUC"],  # 4个指标名称
    figsize=(10, 6),
    fontsize=14
):
    """通用单图绘制函数：输入一组均值+标准差，绘制批次-指标性能图"""
    # 1. 数据格式校验+统一转为二维数组（避免一维输入报错）
    mean_arr = np.atleast_2d(mean_value)
    std_arr = np.atleast_2d(std_value)
    
    # 2. 确定批次数量（X轴）
    n_batches = mean_arr.shape[0]
    batch_x = np.arange(1, n_batches + 1)  # X轴：1,2,3...（批次号）
    
    # 3. 定义4个指标的颜色（固定配色，便于统一对比）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 4. 绘图核心逻辑
    plt.figure(figsize=figsize)
    for i in range(min(len(metrics), mean_arr.shape[1])):  # 防止指标数越界
        # 提取当前指标的均值+标准差
        mean = mean_arr[:, i]
        std = std_arr[:, i]
        
        # 绘制折线+误差带
        plt.plot(batch_x, mean, marker='o', label=metrics[i], color=colors[i], linewidth=2)
        plt.fill_between(batch_x, mean-std, mean+std, alpha=0.2, color=colors[i])
    
    # 5. 图表美化（通用配置）
    plt.title(title, fontsize=fontsize+2)
    plt.xlabel('Batch Number', fontsize=fontsize)
    plt.ylabel('Metric Value', fontsize=fontsize)
    plt.xticks(batch_x, fontsize=fontsize-2)  # X轴刻度匹配批次号
    plt.yticks(fontsize=fontsize-2)
    plt.legend(fontsize=fontsize-2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
