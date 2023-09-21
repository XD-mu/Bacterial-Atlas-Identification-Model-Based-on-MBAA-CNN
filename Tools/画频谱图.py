import openpyxl
import matplotlib.pyplot as plt
import numpy as np

# 打开Excel文件
workbook = openpyxl.load_workbook("金黄色葡萄球菌 (2).xlsx")

# 选择要读取的工作表
sheet = workbook.active

# 初始化空的横坐标和纵坐标列表
x_values = []
y_values = []

# 逐行读取数据
for row in sheet.iter_rows(values_only=True):
    if len(row) == 2:
        x, y = row
        x_values.append(x)
        y_values.append(y)

# 将频谱图分为三段
num_segments = 3
segment_length = len(x_values) // num_segments

# 划分数据为三段
segmented_x = [x_values[i:i + segment_length] for i in range(0, len(x_values), segment_length)]
segmented_y = [y_values[i:i + segment_length] for i in range(0, len(y_values), segment_length)]

# 定义自定义颜色
custom_colors = ['#FFD700', '#FFB6C1','#C2E392' ]

# 绘制每个段的频谱图
plt.figure(figsize=(10, 6))  # 可选，设置图形大小

for i in range(num_segments):
    plt.plot(segmented_x[i], segmented_y[i], label=f'Segment {i + 1}', color=custom_colors[i])

# plt.title("分段频谱图")  # 设置图标题
# plt.legend()  # 添加图例
plt.show()
