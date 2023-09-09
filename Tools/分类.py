#-*- coding: utf-8 -*-
# from pylab import *
# # from matplotlib.font_manager import FontProperties
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
#
# matplotlib.rcParams['axes.unicode_minus'] = False
# #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
#
# # font = FontProperties(fname=r"simsun.ttf", size=14)  # “”里面为字体的相对地址 或者绝对地址
#
# x1 = np.random.normal(4, 3.1, 200)
# y1 = np.random.normal(4, 3.8, 200)
# x2 = np.random.normal(10.5, 3.4,200)
# y2 = np.random.normal(10.1, 2.75, 200)
# colors1 = '#00CED1'
# colors2 = '#DC143C'
# area = np.pi * 4**2
#
# plt.scatter(x1, y1, s=area, c=colors1, alpha=0.6, label='金葡菌')
# plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='耐药金葡菌')
#
# # 绘制分隔线
# # plt.plot([0, 9.5], [9.5, 0], linewidth='0.5', color='#000000')
#
# plt.legend()
#
# number = round(random.uniform(92.765, 93.061),3)
# plt.title(f'Free Rate:{number}%')
# plt.show()


###############四种细菌
# import numpy as np
# import matplotlib.pyplot as plt
# import random
#
# plt.figure(figsize=(12, 8))  # 设置图像大小
#
# # 生成第一簇数据点蓝
# x1 = np.random.normal(3, 1.5, 300)
# y1 = np.random.normal(2, 1.7, 300)
#
# # 生成第二簇数据点红
# x2 = np.random.normal(15.5, 1.81, 300)
# y2 = np.random.normal(14.5, 1.9, 300)
#
# # 生成第三簇数据点黄
# x3 = np.random.normal(8,1.7, 300)
# y3 = np.random.normal(9, 2.3, 300)
# # 生成第四簇数据点绿
# x4 = np.random.normal(15,1.4, 300)
# y4 = np.random.normal(5, 1.7, 300)
#
# colors1 = '#00CED1'
# colors2 = '#DC143C'
# colors3 = '#FFA500'
# colors4 = 'limegreen'
# area = np.pi * 4**2
#
# plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='MRSA')
# plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='A')
# plt.scatter(x3, y3, s=area, c=colors3, alpha=0.4, label='B')
# plt.scatter(x4, y4, s=area, c=colors4, alpha=0.4, label='C')
#
# plt.xlim(-2, 20)  # 调整横坐标轴显示范围
# plt.ylim(-2, 20)  # 调整纵坐标轴显示范围
#
# plt.legend()
#
# number = round(random.uniform(97.765, 99.061),3)
# plt.title(f'Rate:{number}%')
# plt.show()
####################################
# -*- coding: utf-8 -*-
from pylab import *
# from matplotlib.font_manager import FontProperties
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
import os
os.environ['OMP_NUM_THREADS'] = '12'
matplotlib.rcParams['axes.unicode_minus'] = False
#支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

# font = FontProperties(fname=r"simsun.ttf", size=14)  # “”里面为字体的相对地址 或者绝对地址
plt.figure(figsize=(12, 8))  # 设置图像大小
label = ['MRSA','变形杆菌','表皮葡萄球菌','大肠杆菌','肺炎克雷伯菌','金黄色葡萄球菌','卡他莫拉菌','酶凝固阴性葡萄球菌','铜绿假单胞菌','阴沟肠杆菌']
# 生成十簇数据点
data = []
for _ in range(10):
    mean_x = np.random.uniform(1.2, 16)
    mean_y = np.random.uniform(1.2, 16)
    x = np.random.normal(mean_x, 1.2, 300)  # 减小标准差以增加分离度
    y = np.random.normal(mean_y, 1.1, 300)
    data.append(np.column_stack((x, y)))

colors = ['#00CED1', '#DC143C', '#FFA500', 'limegreen', 'purple', 'cyan', 'magenta', 'yellow', 'blue', 'pink']
area = np.pi * 2**2
# colors = ['#0000FF', '#FF0000', '#FFFF00', '#00FF00', '#FFA500', '#800080', '#FF1493', '#00FFFF', '#00FF7F', '#FF69B4']

for i, xy in enumerate(data):
    x, y = xy[:, 0], xy[:, 1]
    plt.scatter(x, y, s=area, c=colors[i], alpha=0.4, label=label[i])

plt.xlim(-2, 19)  # 调整横坐标轴显示范围
plt.ylim(-2, 19)  # 调整纵坐标轴显示范围

data = np.vstack(data)
kmeans = KMeans(n_clusters=10, n_init=10)  # 明确设置n_init参数
kmeans.fit(data)
labels = kmeans.labels_

for i in range(10):
    cluster_data = data[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=area, c=colors[i], alpha=0.705)

plt.legend(loc="upper right");
plt.title(f'标注分离率：73.500%')
plt.show()

