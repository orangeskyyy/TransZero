# import matplotlib.pyplot as plt
# import numpy as np
#
# # 只保留指定的四个数据集
# datasets = ['DBLP', 'CoCS', 'Physics', 'Reddit']
#
# # Inductive - NMI
# qd_gnn_nmi_inductive = [0.22, 0.01, 0.001, 0.001]
# coclep_nmi_inductive = [0.05, 0.15, 0.13, 0.19]
# transzero_ls_nmi_inductive = [0.0663, 0.2278, 0.1304, 0.3131]
# transzero_gs_nmi_inductive = [0.0628, 0.2216, 0.1308, 0.3107]
#
#
#
# # Inductive - JAC
# qd_gnn_jac_inductive = [0.28, 0.32, 0.28, 0.22]
# coclep_jac_inductive = [0.24, 0.20, 0.22, 0.28]
# transzero_ls_jac_inductive = [0.2819, 0.2721, 0.3687, 0.3554]
# transzero_gs_jac_inductive = [0.2764, 0.2743, 0.3691, 0.3527]
#
#
#
# # 柱子宽度
# bar_width = 0.15
# # x轴位置
# r1 = np.arange(len(datasets))
# r2 = [x + bar_width for x in r1]
# r3 = [x + bar_width for x in r2]
# r4 = [x + bar_width for x in r3]
#
# # 设置图片清晰度
# plt.rcParams['figure.dpi'] = 300
# # 设置字体大小
# # 修改全局字体大小
# plt.rcParams.update({'font.size': 12})
# # 设置支持中文的字体，以微软雅黑为例
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# # 解决负号显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False
# # 创建2行3列的子图布局
# fig, axs = plt.subplots(1, 2, figsize=(12, 8))
#
# # 定义绘图函数以减少代码重复
# def plot_bars(ax, data1, data2, data3, data4, title, ylabel):
#     bars_qd_gnn = ax.bar(r1, data1, width=bar_width, edgecolor='black', label='QD-GNN', hatch='//', color='lightblue')
#     bars_transzero_ls = ax.bar(r2, data2, width=bar_width, edgecolor='black', label='BotCS-GNR-LS', hatch='xx', color='lightgreen')
#     bars_coclep = ax.bar(r3, data3, width=bar_width, edgecolor='black', label='COCLEP', color='orange')
#     bars_transzero_gs = ax.bar(r4, data4, width=bar_width, edgecolor='black', label='BotCS-GNR-GS', hatch='xx', color='pink')
#     ax.set_title(title)
#     ax.set_xlabel('数据集', fontsize=14)  # 设置x轴标签字体大小
#     ax.set_ylabel(ylabel, fontsize=14)  # 设置y轴标签字体大小
#     ax.set_xticks([r + bar_width * 1.5 for r in r1], datasets, fontsize=12)  # 设置x轴刻度字体大小
#     ax.tick_params(axis='y', labelsize=12)  # 设置y轴刻度字体大小
#     # 设置图例字体大小
#     ax.legend(fontsize=10)
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#
#     # 找到所有数据中的最小值和最大值
#     all_data = data1 + data2 + data3 + data4
#     min_val = min(all_data)
#     max_val = max(all_data)
#
#     # 根据数据范围设置y轴范围，使柱子高度差更明显
#     y_min = max(0, min_val - 0.05)
#     y_max = max_val + 0.05
#     ax.set_ylim(y_min, y_max)
#
# # 绘制 Inductive - NMI 柱状图
# plot_bars(axs[0, 0], qd_gnn_nmi_inductive, transzero_ls_nmi_inductive, coclep_nmi_inductive, transzero_gs_nmi_inductive, 'Inductive-NMI', 'NMI')
#
# # 绘制 Transductive - NMI 柱状图
# # plot_bars(axs[0, 1], qd_gnn_nmi_transductive, transzero_ls_nmi_transductive, coclep_nmi_transductive, transzero_gs_nmi_transductive, 'Transductive-NMI', 'NMI')
#
# # 绘制 Hybrid - NMI 柱状图
# # plot_bars(axs[1, 1], qd_gnn_nmi_hybrid, transzero_ls_nmi_hybrid, coclep_nmi_hybrid, transzero_gs_nmi_hybrid, 'Hybrid-NMI', 'NMI')
#
# # 绘制 Inductive - JAC 柱状图
# plot_bars(axs[0, 2], qd_gnn_jac_inductive, transzero_ls_jac_inductive, coclep_jac_inductive, transzero_gs_jac_inductive, 'Inductive-JAC', 'JAC')
#
# # 绘制 Transductive - JAC 柱状图
# # plot_bars(axs[1, 0], qd_gnn_jac_transductive, transzero_ls_jac_transductive, coclep_jac_transductive, transzero_gs_jac_transductive, 'Transductive-JAC', 'JAC')
#
# # 绘制 Hybrid - JAC 柱状图
# # plot_bars(axs[1, 2], qd_gnn_jac_hybrid, transzero_ls_jac_hybrid, coclep_jac_hybrid, transzero_gs_jac_hybrid, 'Hybrid-JAC', 'JAC')
#
# # 调整子图布局
# plt.tight_layout()
# # 显示图形
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 全局设置
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 数据准备
datasets = ['DBLP', 'CoCS', 'Physics', 'Reddit']

# NMI 数据（Inductive）
qd_gnn_nmi = [0.22, 0.01, 0.001, 0.001]
coclep_nmi = [0.05, 0.15, 0.13, 0.19]
transzero_ls_nmi = [0.0663, 0.2278, 0.1304, 0.3131]
transzero_gs_nmi = [0.0628, 0.2216, 0.1308, 0.3107]

# JAC 数据（Inductive）
qd_gnn_jac = [0.28, 0.32, 0.28, 0.22]
coclep_jac = [0.24, 0.20, 0.22, 0.28]
transzero_ls_jac = [0.2819, 0.2721, 0.3687, 0.3554]
transzero_gs_jac = [0.2764, 0.2743, 0.3691, 0.3527]

# 柱状图参数
bar_width = 0.15
x = np.arange(len(datasets))

# ============================================
# 1. 绘制NMI独立图表
plt.figure(figsize=(10, 6))

# 绘制柱状图
bars1 = plt.bar(x - 1.5*bar_width, qd_gnn_nmi, width=bar_width,
               label='QD-GNN', hatch='//', color='lightblue')
bars2 = plt.bar(x - 0.5*bar_width, coclep_nmi, width=bar_width,
               label='CocLEP', hatch='xx', color='lightgreen')
bars3 = plt.bar(x + 0.5*bar_width, transzero_ls_nmi, width=bar_width,
               label='BotCS-GNR-LS', color='orange')
bars4 = plt.bar(x + 1.5*bar_width, transzero_gs_nmi, width=bar_width,
               label='BotCS-GNR-GS', hatch='..', color='pink')

# 添加数据标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# 图表装饰
# plt.title('各方法在不同数据集上的NMI指标对比', fontsize=14)
plt.xlabel('数据集', fontsize=16)
plt.ylabel('NMI', fontsize=16)
plt.xticks(x, datasets)
plt.legend(loc='upper left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.ylim(0, 0.35)  # 设置Y轴范围

plt.tight_layout()
plt.savefig('5.6(a)NMI_对比.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2. 绘制JAC独立图表
plt.figure(figsize=(10, 6))

# 绘制柱状图
bars1 = plt.bar(x - 1.5*bar_width, qd_gnn_jac, width=bar_width,
               label='QD-GNN', hatch='//', color='lightblue')
bars2 = plt.bar(x - 0.5*bar_width, coclep_jac, width=bar_width,
               label='CocLEP', hatch='xx', color='lightgreen')
bars3 = plt.bar(x + 0.5*bar_width, transzero_ls_jac, width=bar_width,
               label='BotCS-GNR-LS', color='orange')
bars4 = plt.bar(x + 1.5*bar_width, transzero_gs_jac, width=bar_width,
               label='BotCS-GNR-GS', hatch='..', color='pink')

# 添加数据标签
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# 图表装饰
# plt.title('各方法在不同数据集上的JAC指标对比', fontsize=14)
plt.xlabel('数据集', fontsize=16)
plt.ylabel('JAC', fontsize=16)
plt.xticks(x, datasets)
plt.legend(loc='upper left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.ylim(0, 0.45)  # 设置Y轴范围

plt.tight_layout()
plt.savefig('5.6(b)JAC_对比.png', dpi=300, bbox_inches='tight')
plt.show()