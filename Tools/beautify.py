import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 自定义颜色映射
def custom_colormap():
    white = np.array([1, 1, 1])  # 白色
    red = np.array([1, 0, 0])    # 红色
    blue = np.array([0, 0, 1])   # 蓝色

    # 创建颜色映射
    colors = [white]

    # 增加色度的数量
    num_steps = 50  # 调整此值以控制颜色变化的数量

    for i in range(num_steps):
        # 在白色和红色之间线性插值，以增加色度
        fraction_red = i / (num_steps - 1)
        interpolated_color_red = (1 - fraction_red) * white + fraction_red * red
        colors.append(interpolated_color_red)
    reversed_color = colors[::-1]
    # 添加反方向的插值，从红色到蓝色
    for i in range(num_steps):
        fraction_blue = i / (num_steps - 1)
        interpolated_color_blue = (1 - fraction_blue) * white + fraction_blue * blue
        reversed_color.append(interpolated_color_blue)
    color=reversed_color[::-1]
    n = len(color)
    cmap_data = np.ones((n, 4))

    for i, ca in enumerate(color):
        cmap_data[i, :3] = ca
        cmap_data[i, 3] = i / (n - 1)

    return mcolors.ListedColormap(cmap_data, name='custom_colormap')
# training
standard_array = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                            5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9])

# Label一维数组，与标准数组形状相同
label_array = np.array([0, 0, 2, 0, 0, 1, 3, 1, 2, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 3, 4, 4, 4,
                        5, 5, 5, 4, 5, 6, 6, 6, 6, 6, 7, 7, 7, 6, 7, 8, 8, 5, 8, 8, 9, 9, 9, 7, 9])

# 未标记的一维数组
unlabel_array = np.array([0, 0, 0, 3, 0, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 3, 1, 3, 3, 3, 4, 4, 4, 4, 4,
                          2, 5, 2, 3, 5, 5, 6, 6, 6, 6, 7, 6, 7, 7, 7, 8, 5, 8, 8, 8, 9, 9, 4, 9, 9])
# testing
# standard_array = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
#
# )
#
# # Label一维数组，与标准数组形状相同
# label_array = np.array([0, 1, 0, 1, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 6, 7, 8, 8, 7, 9, 9, 9]
#
# )
#
# # 未标记的一维数组
# unlabel_array = np.array([1, 0, 1, 1, 0, 1, 2, 2, 2, 3, 2, 3, 4, 2, 4, 5, 5, 5, 4, 6, 6, 7, 9, 7, 8, 8, 5, 9, 9, 9]
#
# )
# 计算插值差值
interpolated_values1 = standard_array - label_array
interpolated_values2 = standard_array - unlabel_array
# 创建颜色映射
cmap = custom_colormap()

# 控制图的高度
fig, ax = plt.subplots(figsize=(30, 4))

# 将两行数据合并成一个数组
combined_data = np.vstack([interpolated_values1,interpolated_values2])

# 创建渐变方格图
img = ax.imshow(combined_data, cmap=custom_colormap(), aspect=2.5, extent=[0, len(standard_array), 0, 2])


# 隐藏坐标轴及其文字
ax.axis('off')

# 添加左侧文本标签
# for i, label in enumerate(['Label', 'UnLabel']):
#     ax.text(-0.2, i + 0.5, label, ha='right', va='center', fontsize=9)

ax.hlines(1, 0, len(standard_array), color='gray', linestyle='--', linewidth=0.8)


# 添加 colorbar，并将其放置在右下角，缩放为0.3，并水平放置
cbar = plt.colorbar(img, ax=ax, label='Interpolated Values', orientation='vertical', pad=0.05, shrink=0.5)
# 调整 colorbar 标签的位置
# cbar.ax.xaxis.set_label_position('bottom')
# cbar.ax.set_xlabel('Interpolated Values', labelpad=10, fontsize=9)

plt.show()





# 下方代码为绘制Actual/label_predict/unlabel_predict的标注显示结果表示
# import matplotlib.pyplot as plt
# import numpy as np
#
# # training
# # standard_array = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
# #                             5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9])
# #
# # # Label一维数组，与标准数组形状相同
# # label_array = np.array([0, 0, 2, 0, 0, 1, 3, 1, 2, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 3, 4, 4, 4,
# #                         5, 5, 5, 4, 5, 6, 6, 6, 6, 6, 7, 7, 7, 6, 7, 8, 8, 5, 8, 8, 9, 9, 9, 7, 9])
# #
# # # 未标记的一维数组
# # unlabel_array = np.array([0, 0, 0, 3, 0, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 3, 1, 3, 3, 3, 4, 4, 4, 4, 4,
# #                           2, 5, 2, 3, 5, 5, 6, 6, 6, 6, 7, 6, 7, 7, 7, 8, 5, 8, 8, 8, 9, 9, 4, 9, 9])
# # testing
# # standard_array = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
# #
# # )
# #
# # # Label一维数组，与标准数组形状相同
# # label_array = np.array([0, 1, 0, 1, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 6, 7, 8, 8, 7, 9, 9, 9]
# #
# # )
# #
# # # 未标记的一维数组
# # unlabel_array = np.array([1, 0, 1, 1, 0, 1, 2, 2, 2, 3, 2, 3, 4, 2, 4, 5, 5, 5, 4, 6, 6, 7, 9, 7, 8, 8, 5, 9, 9, 9]
# #
# # )
# # 创建颜色映射
# color_mapping = {
#     0: "#9B3A4D",
#     1: "#394A92",
#     2: "#E2AE79",
#     3: "#0EEBB0",
#     4: "#D0DCAA",
#     5: "#566CA5",
#     6: "#8CBDA7",
#     7: "#70A0AC",
#     8: "#68AC57",
#     9: "#497EB2"
# }
#
# # 创建图形
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # 绘制standard数组
# for i, color_idx in enumerate(standard_array):
#     ax.fill_betweenx([0, 1], i, i + 1, color=color_mapping[color_idx], edgecolor='black')
#
# # 绘制label数组
# for i, color_idx in enumerate(label_array):
#     ax.fill_betweenx([1, 2], i, i + 1, color=color_mapping[color_idx], edgecolor='black')
#
# # 绘制unlabel数组
# for i, color_idx in enumerate(unlabel_array):
#     ax.fill_betweenx([2, 3], i, i + 1, color=color_mapping[color_idx], edgecolor='black')
#
# # 设置x轴范围和标签
# ax.set_xlim(0, max(len(standard_array), len(label_array), len(unlabel_array)))
# ax.set_xticks([])
#
# # 去掉坐标轴及其文字
# ax.axis('off')
#
# # 旋转90度
# ax.set_aspect(1)
#
# # 显示图形
# plt.show()
