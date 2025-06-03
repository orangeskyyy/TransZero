import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
def nmi_jac():

    # 数据准备
    datasets = ['Cornell', 'Cora', 'Citeseer', 'Photo']

    # Inductive - NMI
    qd_gnn_nmi = [0.0458, 0.0445, 0.0128, 0.0675]
    coclep_nmi = [0.0089, 0.0512, 0.0131, 0.1258]
    global_search_vrc_nmi = [0.0168, 0.0623, 0.0224, 0.2286]
    global_search_sli_nmi = [0.0421, 0.1076, 0.1053, 0.2775]

    # Inductive - JAC
    qd_gnn_jac = [0.0236, 0.0021, 0.0158, 0.0183]
    coclep_jac = [0.1786, 0.1858, 0.1862, 0.2832]
    global_search_vrc_jac = [0.1964, 0.2395, 0.1885, 0.3883]
    global_search_sli_jac = [0.2378, 0.2940, 0.2951, 0.4276]

    # 柱状图参数
    bar_width = 0.15
    x = np.arange(len(datasets))

    # 自定义函数用于格式化Y轴标签
    def percent_formatter(x, pos):
        return f'{x:.2f}'

    # 自定义颜色方案
    colors = {
        'qd_gnn': '#4a86e8',  # 蓝色系
        'coclep': '#6aa84f',  # 绿色系
        'vrc': '#f1c232',  # 黄色系
        'sli': '#f6b26b'  # 橙色系
    }

    # 自定义图案填充
    hatch_patterns = ['//', 'xx', '..', 'oo']

    # 设置图表风格
    plt.style.use('seaborn-whitegrid')

    # 创建画布和子图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ============================================
    # 1. 绘制NMI图表
    ax1 = axes[0]

    # 绘制柱状图
    bars1 = ax1.bar(x - 1.5 * bar_width, qd_gnn_nmi, width=bar_width,
                    label='QD-GNN', hatch=hatch_patterns[0], color=colors['qd_gnn'], edgecolor='black')
    bars2 = ax1.bar(x - 0.5 * bar_width, coclep_nmi, width=bar_width,
                    label='CocLEP', hatch=hatch_patterns[1], color=colors['coclep'], edgecolor='black')
    bars3 = ax1.bar(x + 0.5 * bar_width, global_search_vrc_nmi, width=bar_width,
                    label='BotCS-GNR-VRC', hatch=hatch_patterns[2], color=colors['vrc'], edgecolor='black')
    bars4 = ax1.bar(x + 1.5 * bar_width, global_search_sli_nmi, width=bar_width,
                    label='BotCS-GNR-Sli', hatch=hatch_patterns[3], color=colors['sli'], edgecolor='black')

    # 添加数据标签
    def add_labels(ax, bars, decimal_places=4):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                    f'{height:.{decimal_places}f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)

    add_labels(ax1, bars1)
    add_labels(ax1, bars2)
    add_labels(ax1, bars3)
    add_labels(ax1, bars4)

    # 图表装饰
    ax1.set_title('各方法在不同数据集上的NMI指标对比', fontsize=16, pad=15)
    ax1.set_xlabel('数据集', fontsize=16)
    ax1.set_ylabel('NMI值', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=12)
    ax1.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.9)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, max(global_search_sli_nmi) * 1.15)  # 动态设置Y轴范围

    # 设置Y轴格式
    ax1.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

    # 添加网格线到柱状图背后
    ax1.set_axisbelow(True)

    # ============================================
    # 2. 绘制JAC图表
    ax2 = axes[1]

    # 绘制柱状图
    bars1 = ax2.bar(x - 1.5 * bar_width, qd_gnn_jac, width=bar_width,
                    label='QD-GNN', hatch=hatch_patterns[0], color=colors['qd_gnn'], edgecolor='black')
    bars2 = ax2.bar(x - 0.5 * bar_width, coclep_jac, width=bar_width,
                    label='CocLEP', hatch=hatch_patterns[1], color=colors['coclep'], edgecolor='black')
    bars3 = ax2.bar(x + 0.5 * bar_width, global_search_vrc_jac, width=bar_width,
                    label='BotCS-GNR-VRC', hatch=hatch_patterns[2], color=colors['vrc'], edgecolor='black')
    bars4 = ax2.bar(x + 1.5 * bar_width, global_search_sli_jac, width=bar_width,
                    label='BotCS-GNR-Sli', hatch=hatch_patterns[3], color=colors['sli'], edgecolor='black')

    # 添加数据标签
    add_labels(ax2, bars1)
    add_labels(ax2, bars2)
    add_labels(ax2, bars3)
    add_labels(ax2, bars4)

    # 图表装饰
    ax2.set_title('各方法在不同数据集上的JAC指标对比', fontsize=16, pad=15)
    ax2.set_xlabel('数据集', fontsize=16)
    ax2.set_ylabel('JAC值', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=12)
    ax2.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.9)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_ylim(0, max(global_search_sli_jac) * 1.15)  # 动态设置Y轴范围

    # 设置Y轴格式
    ax2.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

    # 添加网格线到柱状图背后
    ax2.set_axisbelow(True)

    # 调整子图布局
    plt.tight_layout(pad=3.0)

    # 保存图片
    plt.savefig('指标对比图.png', dpi=600, bbox_inches='tight')

    # 显示图形
    plt.show()

def t_parameters():
    # 参数值
    tau = [0.1, 0.3, 0.5, 0.7, 0.9]

    # 四个数据集对应的F1值
    f1_values_dataset1 = [0.3879, 0.3920, 0.3842, 0.2451, 0.1113]  # Cornell
    f1_values_dataset2 = [0.3864, 0.4273, 0.4544, 0.4265, 0.0832]  # Cora
    f1_values_dataset3 = [0.4729, 0.4838, 0.4557, 0.3788, 0.1394]  # Citeseer
    f1_values_dataset4 = [0.4569, 0.5182, 0.5990, 0.6274, 0.4810]  # Photo

    # 定义四种不同的颜色
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33F6']  # 红、绿、蓝、紫

    # 创建2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 设置坐标轴标签和标题的字体大小
    label_fontsize = 14
    title_fontsize = 16

    # 绘制第一个子图（左上角）
    axs[0, 0].plot(tau, f1_values_dataset1, marker='o', color=colors[0], linewidth=2, markersize=8)
    axs[0, 0].set_xlabel('$\\tau$', fontsize=label_fontsize)
    axs[0, 0].set_ylabel('F1值', fontsize=label_fontsize)
    axs[0, 0].set_title('Cornell', fontsize=title_fontsize)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=12)

    # 绘制第二个子图（右上角）
    axs[0, 1].plot(tau, f1_values_dataset2, marker='o', color=colors[1], linewidth=2, markersize=8)
    axs[0, 1].set_xlabel('$\\tau$', fontsize=label_fontsize)
    axs[0, 1].set_ylabel('F1值', fontsize=label_fontsize)
    axs[0, 1].set_title('Cora', fontsize=title_fontsize)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=12)

    # 绘制第三个子图（左下角）
    axs[1, 0].plot(tau, f1_values_dataset3, marker='o', color=colors[2], linewidth=2, markersize=8)
    axs[1, 0].set_xlabel('$\\tau$', fontsize=label_fontsize)
    axs[1, 0].set_ylabel('F1值', fontsize=label_fontsize)
    axs[1, 0].set_title('Citeseer', fontsize=title_fontsize)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=12)

    # 绘制第四个子图（右下角）
    axs[1, 1].plot(tau, f1_values_dataset4, marker='o', color=colors[3], linewidth=2, markersize=8)
    axs[1, 1].set_xlabel('$\\tau$', fontsize=label_fontsize)
    axs[1, 1].set_ylabel('F1值', fontsize=label_fontsize)
    axs[1, 1].set_title('Photo', fontsize=title_fontsize)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=12)

    # 添加网格线
    for ax in axs.flat:
        ax.grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.tight_layout(pad=3.0)  # 增加子图间距

    # 显示图形
    plt.show()

if __name__ == '__main__':
    # 全局设置
    plt.rcParams.update({'font.size': 12})
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 尝试加载字体
    try:
        plt.rcParams["font.family"] = ["SimHei"]
    except:
        try:
            plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
        except:
            try:
                plt.rcParams["font.family"] = ["Heiti TC"]
            except:
                plt.rcParams["font.family"] = ["sans-serif"]
                print("警告: 未找到中文字体，图表中的中文可能无法正确显示。")
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    nmi_jac()
    # t_parameters()