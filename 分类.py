import matplotlib.pyplot as plt
import numpy as np

def social_data_aug():
    # 第一张表数据
    data1 = np.array([
        [0.96604216987854, 0.9710982441902161, 0.972530297537231, 0.9722864031791687, 0.9716874361038208],
        [0.958516538141358, 0.9630523992005924, 0.9667923450469971, 0.9651851058006287, 0.964526355266571],
        [0.9537907838821412, 0.9596207141876221, 0.9535393714904785, 0.9491798281669617, 0.9552224278450012],
        [0.9635984783392212, 0.9676963090866066, 0.9805359840390666, 0.9818332195281982, 0.9740968364595764]
    ])
    labels1 = ['acc', 'f1', 'precision', 'recall']

    # 第二张表数据
    data2 = np.array([
        [0.96988376059532166, 0.9691780805587769, 0.969525933265686, 0.9702176465188044, 0.9667811989784240],
        [0.9622270464897156, 0.9620493665504456, 0.9628956371906163, 0.9639092683792114, 0.9582678079605103],
        [0.9537566995657023, 0.9553011655087495, 0.940833689266251, 0.9456916451441463, 0.9406938917488865],
        [0.9724092483520508, 0.9692410286883944, 0.9788736104956532, 0.9821265695640806, 0.9775122040499511]
    ])
    labels2 = ['acc', 'f1', 'precision', 'recall']

    # 第三张表数据
    data3 = np.array([
        [0.9732868671417236, 0.9732868671417236, 0.9756662845611572, 0.9720605611801147, 0.9695906639099120],
        [0.9681899547576904, 0.9670879244404882, 0.9702296807403566, 0.966235046183472, 0.9637730121612549],
        [0.9548247566550968, 0.9476581128588466, 0.9508720184064658, 0.9586335658340623, 0.9264040214734558],
        [0.9829120635986232, 0.9880015850671139, 0.9892062861045532, 0.9737901091575623, 0.9785605372055188]
    ])
    labels3 = ['acc', 'f1', 'precision', 'recall']

    # 横坐标
    x = ['0.1', '0.3', '0.5', '0.7', '0.9']

    # 创建子图
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制第一个子图
    for i in range(len(data1)):
        if i == 0:
            axs[0].plot(x, data1[i], label=labels1[i],marker='o',linestyle='--')
        elif i == 1:
            axs[0].plot(x, data1[i], label=labels1[i],marker='s',linestyle='--')
        elif i == 2:
            axs[0].plot(x, data1[i], label=labels1[i],marker='v',linestyle='--')
        else:
            axs[0].plot(x, data1[i], label=labels1[i],marker='^',linestyle='--')
    axs[0].set_title('(a)边移除pe超参')
    axs[0].set_xlabel('概率')
    axs[0].set_ylabel('指标计算值')
    axs[0].set_ylim(0.90,0.985)
    axs[0].legend()

    # 绘制第二个子图
    for i in range(len(data2)):
        if i == 0:
            axs[1].plot(x, data2[i], label=labels2[i],marker='o',linestyle='--')
        elif i == 1:
            axs[1].plot(x, data2[i], label=labels2[i],marker='s',linestyle='--')
        elif i == 2:
            axs[1].plot(x, data2[i], label=labels2[i],marker='v',linestyle='--')
        else:
            axs[1].plot(x, data2[i], label=labels2[i],marker='^',linestyle='--')
    axs[1].set_title('(b)边增加pa超参')
    axs[1].set_xlabel('概率')
    axs[1].set_ylabel('指标计算值')
    axs[1].set_ylim(0.90,0.985)
    axs[1].legend()

    # 绘制第三个子图
    for i in range(len(data3)):
        if i == 0:
            axs[2].plot(x, data3[i], label=labels2[i], marker='o', linestyle='--')
        elif i == 1:
            axs[2].plot(x, data3[i], label=labels2[i], marker='s', linestyle='--')
        elif i == 2:
            axs[2].plot(x, data3[i], label=labels2[i], marker='v', linestyle='--')
        else:
            axs[2].plot(x, data3[i], label=labels2[i], marker='^', linestyle='--')

    axs[2].set_title('(c)属性掩盖pf超参')
    axs[2].set_xlabel('概率')
    axs[2].set_ylabel('指标计算值')
    axs[2].set_ylim(0.90,0.99)
    axs[2].legend()

def transformer_heads1():
    # Transformer头数
    trans_heads = [2, 4, 6]
    # acc值
    time = [12.4, 15.5, 21.3]

    # 绘制柱状图
    bars = plt.bar(trans_heads, time)
    # 绘制折线图
    # plt.plot(trans_heads, time,  color='red', linestyle='--')

    # 在每个柱形上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height}',
                 ha='center', va='bottom')
    # 添加标题
    # plt.title('不同Transformer头数下的模型运行时间')
    # 添加x轴标签
    plt.xlabel('Transformer头数')
    # 添加y轴标签
    plt.ylabel('时间（分钟）')
    # 添加图例
    plt.legend()

    # 添加图例
    plt.legend()

def transformer_heads2():
    # 数据
    trans_heads = [2, 4, 6]
    acc = [0.9738041162490845, 0.9747416973114014, 0.9722864031791681]
    f1 = [0.9674229621887207, 0.968894064422421, 0.967022716999054]
    precision = [0.9580955949336165, 0.95611572265625, 0.9428711808408349]
    recall = [0.9773046709734742, 0.9821502566356758, 0.9866853563561389]

    # 创建4个子图
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.ravel()

    # 定义柱子宽度
    bar_width = 0.8

    # 绘制acc子图（柱状图）
    bars_acc = axs[0].bar(trans_heads, acc, width=bar_width)
    axs[0].set_title('acc')
    axs[0].set_xlabel('Transformer头数')
    axs[0].set_ylim(0.97, 0.975)
    # 在柱子上添加数值
    for bar in bars_acc:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, height,
                    '{:.4f}'.format(height), ha='center', va='bottom')

    # 绘制f1子图（柱状图）
    bars_f1 = axs[1].bar(trans_heads, f1, width=bar_width)
    axs[1].set_title('f1')
    axs[1].set_xlabel('Transformer头数')
    axs[1].set_ylim(0.965, 0.97)
    # 在柱子上添加数值
    for bar in bars_f1:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, height,
                    '{:.4f}'.format(height), ha='center', va='bottom')

    # 绘制precision子图（柱状图）
    bars_precision = axs[2].bar(trans_heads, precision, width=bar_width)
    axs[2].set_title('precision')
    axs[2].set_xlabel('Transformer头数')
    axs[2].set_ylim(0.94, 0.96)
    # 在柱子上添加数值
    for bar in bars_precision:
        height = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width() / 2, height,
                    '{:.4f}'.format(height), ha='center', va='bottom')

    # 绘制recall子图（柱状图）
    bars_recall = axs[3].bar(trans_heads, recall, width=bar_width)
    axs[3].set_title('recall')
    axs[3].set_xlabel('Transformer头数')
    axs[3].set_ylim(0.97, 0.99)
    # 在柱子上添加数值
    for bar in bars_recall:
        height = bar.get_height()
        axs[3].text(bar.get_x() + bar.get_width() / 2, height,
                    '{:.4f}'.format(height), ha='center', va='bottom')

    # 调整子图之间的间距
    plt.tight_layout()


def semi_alpha():
    # 数据
    # 横坐标数据
    a_values = [0.3, 0.5, 0.7]

    # 不同指标的数据
    acc = [0.97412864037375, 0.9670079350414195, 0.9732868671417236]
    f1 = [0.9670727529620025, 0.9603458046913147, 0.9741215993881226]
    precision = [0.9521687226838604, 0.9396985617520323, 0.9623628372337952]
    recall = [0.982566693100758, 0.9852636904699696, 0.9839393944404171]

    # 绘制折线图
    plt.plot(a_values, acc, label='acc')
    plt.plot(a_values, f1, label='f1')
    plt.plot(a_values, precision, label='precision')
    plt.plot(a_values, recall, label='recall')


    # 添加标签和标题
    plt.xlabel('半监督损失函数权重alpha')
    plt.ylabel('指标值')
    plt.title('不同权重alpha下的指标表现')
    plt.legend()

def semi_alpha2():
    # 数据
    alpha = [0.3, 0.4, 0.5, 0.6, 0.7]
    accuracy = [0.9557, 0.9547, 0.9579, 0.9563, 0.9552]
    f1_score = [0.8921, 0.8894, 0.8962, 0.8925, 0.8882]
    precision = [0.8616, 0.8611, 0.8775, 0.8703, 0.8731]
    recall = [0.9250, 0.9198, 0.9160, 0.9163, 0.9040]

    # 绘制折线图
    plt.plot(alpha, accuracy, label='acc', marker='o',linestyle='--')
    plt.plot(alpha, f1_score, label='f1', marker='s',linestyle='--')
    plt.plot(alpha, precision, label='precision', marker='v',linestyle='--')
    plt.plot(alpha, recall, label='recall', marker='^',linestyle='--')


    # 添加标签和标题
    plt.xlabel('α')
    plt.ylabel('指标值')
    plt.title('不同α值下的指标表现')

    # 添加图例
    plt.legend()

def ablation():
    # 模型名称
    models = ["GCN w/o D-RGT", "GAT w/o D-RGT", "w/o data augment", "w/o semi-supervised", "Ours"]

    # TwiBot-20数据集数据
    twi_acc = [86.39, 87.77, 87.95, 86.12, 88.29]
    twi_f1 = [87.91, 88.85, 89.12, 87.81, 89.42]
    twi_p = [84.58, 85.73, 84.94, 83.88, 87.72]
    twi_r = [91.57, 90.92, 93.92, 90.14, 91.28]

    # MGTAB-22数据集数据
    mgtab_acc = [95.52, 95.52, 97.25, 94.62, 97.35]
    mgtab_f1 = [88.17, 91.86, 92.75, 86.89, 92.95]
    mgtab_p = [84.45, 92.16, 91.92, 84.12, 92.66]
    mgtab_r = [87.91, 91.58, 92.67, 89.66, 93.26]

    # 适当调大柱子宽度
    bar_width = 0.21

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # 蓝色系颜色列表
    blue_colors = ['#084594', '#2171b5', '#4292c6', '#6baed6']
    # 不同的条纹样式列表
    hatches = ['/', '\\', '|', '-']

    # 绘制TwiBot-20数据集的柱状图
    r1_twi = np.arange(len(models))
    r2_twi = [x + bar_width for x in r1_twi]
    r3_twi = [x + bar_width for x in r2_twi]
    r4_twi = [x + bar_width for x in r3_twi]

    bars_twi_acc = ax1.bar(r1_twi, twi_acc, width=bar_width, label='Acc', color=blue_colors[0], hatch=hatches[0])
    bars_twi_f1 = ax1.bar(r2_twi, twi_f1, width=bar_width, label='F1', color=blue_colors[1], hatch=hatches[1])
    bars_twi_p = ax1.bar(r3_twi, twi_p, width=bar_width, label='P', color=blue_colors[2], hatch=hatches[2])
    bars_twi_r = ax1.bar(r4_twi, twi_r, width=bar_width, label='R', color=blue_colors[3], hatch=hatches[3])

    # 在TwiBot-20的柱子上添加数值
    def add_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    add_labels(ax1, bars_twi_acc)
    add_labels(ax1, bars_twi_f1)
    add_labels(ax1, bars_twi_p)
    add_labels(ax1, bars_twi_r)

    ax1.set_xlabel('模型名称')
    ax1.set_ylabel('指标数值')
    ax1.set_title('TwiBot-20')
    ax1.set_xticks([r + bar_width * 1.5 for r in r1_twi])
    ax1.set_xticklabels(models)
    # 调整TwiBot-20子图的纵坐标范围
    twi_min = min(min(twi_acc), min(twi_f1), min(twi_p), min(twi_r))
    twi_max = max(max(twi_acc), max(twi_f1), max(twi_p), max(twi_r))
    ax1.set_ylim(twi_min - 1, twi_max + 1)
    # 调整TwiBot-20子图的legend位置
    ax1.legend(loc='lower left')

    # 绘制MGTAB-22数据集的柱状图
    r1_mgtab = np.arange(len(models))
    r2_mgtab = [x + bar_width for x in r1_mgtab]
    r3_mgtab = [x + bar_width for x in r2_mgtab]
    r4_mgtab = [x + bar_width for x in r3_mgtab]

    bars_mgtab_acc = ax2.bar(r1_mgtab, mgtab_acc, width=bar_width, label='Acc', color=blue_colors[0], hatch=hatches[0])
    bars_mgtab_f1 = ax2.bar(r2_mgtab, mgtab_f1, width=bar_width, label='F1', color=blue_colors[1], hatch=hatches[1])
    bars_mgtab_p = ax2.bar(r3_mgtab, mgtab_p, width=bar_width, label='P', color=blue_colors[2], hatch=hatches[2])
    bars_mgtab_r = ax2.bar(r4_mgtab, mgtab_r, width=bar_width, label='R', color=blue_colors[3], hatch=hatches[3])

    # 在MGTAB-22的柱子上添加数值
    add_labels(ax2, bars_mgtab_acc)
    add_labels(ax2, bars_mgtab_f1)
    add_labels(ax2, bars_mgtab_p)
    add_labels(ax2, bars_mgtab_r)

    ax2.set_xlabel('模型名称')
    ax2.set_ylabel('指标数值')
    ax2.set_title('MGTAB-22')
    ax2.set_xticks([r + bar_width * 1.5 for r in r1_mgtab])
    ax2.set_xticklabels(models)
    # 调整MGTAB-22子图的纵坐标范围
    mgtab_min = min(min(mgtab_acc), min(mgtab_f1), min(mgtab_p), min(mgtab_r))
    mgtab_max = max(max(mgtab_acc), max(mgtab_f1), max(mgtab_p), max(mgtab_r))
    ax2.set_ylim(mgtab_min - 1, mgtab_max + 1)
    # 调整MGTAB-22子图的legend位置
    ax2.legend(loc='lower left')

    # 调整子图布局
    plt.tight_layout()


if __name__ == '__main__':
    # 超参实验绘图
    # social_data_aug()
    transformer_heads1()
    # transformer_heads2()
    # semi_alpha2()
    # 消融实验绘图
    # ablation()
    # 设置支持中文的字体，以微软雅黑为例
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    # 调整子图布局
    plt.tight_layout()

    # 显示图形
    plt.show()