import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# 全局样式设置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 18,
    'axes.labelcolor': '#2d2d2d',
    'axes.titlesize': 14
})

# 模型标签
categories = ['GC0.6', 'GCL', 'GC1.7', 'MN4', 'GC4']
metrics = ['Comprehensiveness', 'Diversity', 'Empowerment', 'Overall']

# multi
data_matrices = {
    'Comprehensiveness': np.array([
        [50, 5, 1, 2, 0],
        [95, 50, 32, 29, 17],
        [99, 68, 50, 32, 3],
        [98, 71, 68, 50, 24],
        [100, 83, 97, 76, 50]
    ]),
    'Diversity': np.array([
        [50, 8, 1, 2, 0],
        [92, 50, 6, 5, 2],
        [99, 94, 50, 25, 19],
        [99, 95, 75, 50, 32],
        [100, 98, 81, 68, 50]
    ]),
    'Empowerment': np.array([
        [50, 10, 2, 2, 1],
        [90, 50, 17, 15, 10],
        [98, 83, 50, 39, 26],
        [98, 85, 61, 50, 38],
        [99, 90, 74, 62, 50]
    ]),
    'Overall': np.array([
        [50, 5, 1, 0, 0],
        [95, 50, 18, 12, 8],
        [99, 82, 50, 28, 10],
        [100, 88, 72, 50, 28],
        [100, 91, 90, 72, 50]
    ])
}

# mix
# 替换为新提供的四组矩阵数据
# data_matrices = {
#     'Comprehensiveness': np.array([
#         [50, 6, 4, 3, 0],
#         [94, 50, 27, 22, 19],
#         [96, 73, 50, 30, 6],
#         [97, 78, 70, 50, 31],
#         [100, 81, 94, 69, 50]
#     ]),
#     'Diversity': np.array([
#         [50, 7, 1, 1, 1],
#         [93, 50, 6, 6, 7],
#         [99, 94, 50, 32, 30],
#         [99, 94, 68, 50, 36],
#         [99, 93, 70, 64, 50]
#     ]),
#     'Empowerment': np.array([
#         [50, 11, 0, 2, 0],
#         [89, 50, 19, 14, 11],
#         [100, 81, 50, 35, 23],
#         [98, 86, 65, 50, 35],
#         [100, 89, 77, 65, 50]
#     ]),
#     'Overall': np.array([
#         [50, 7, 0, 1, 0],
#         [93, 50, 18, 13, 10],
#         [100, 82, 50, 31, 16],
#         [99, 87, 69, 50, 32],
#         [100, 90, 84, 68, 50]
#     ])
# }


# 蓝绿色渐变色板
color_gradient = ["#e8f4f8", "#a7d7e8", "#5fb3c8", "#2a7a8c", "#004455"]
cmap = LinearSegmentedColormap.from_list('aqua_blue', color_gradient, N=256)

# 创建画布
fig = plt.figure(figsize=(20, 6))
gs = GridSpec(1, 4, figure=fig, wspace=0.35, hspace=0.05)

# 绘图
for idx, metric in enumerate(metrics):
    ax = fig.add_subplot(gs[idx])
    matrix = data_matrices[metric]
    
    # 热力图
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100)
    
    # 数值标注
    for i in range(5):
        for j in range(5):
            val = matrix[i, j]
            text_color = 'white' if val > 60 else '#2d2d2d'
            ax.text(j, i, f"{val}", ha='center', va='center', 
                    color=text_color, fontsize=26)
    
    # 设置上方的模型名
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(categories, rotation=15, ha='center', fontsize=20)
    ax.set_yticklabels(categories, fontsize=20)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(bottom=False, labelbottom=False, top=True)
    ax.tick_params(axis='both', which='both', length=0)

    # 子图下方标签：靠近热力图底部
    ax.text(0.5, -0.10, f'({chr(97+idx)}) {metric}', transform=ax.transAxes,
            fontsize= 26, fontweight='semibold', color='#1a1a1a',
            ha='center', va='center')

    # 边框
    for spine in ax.spines.values():
        spine.set_color('#b0b0b0')
        spine.set_linewidth(0.8)

# 布局优化
plt.subplots_adjust(left=0.06, right=0.99,top=0.91, bottom=0.22)
plt.show()
