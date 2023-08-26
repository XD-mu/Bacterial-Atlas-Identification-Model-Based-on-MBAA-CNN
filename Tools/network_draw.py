from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

from keras.regularizers import l1, l2
input_shape = (1200, 2)
num_classes = 6

model = Sequential()
model.add(Conv1D(512, 3, activation='relu', input_shape=input_shape, kernel_regularizer=l1(0.01)))
model.add(Dropout(0.01))  # 预防过拟合
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(256, 3, activation='relu', input_shape=input_shape, kernel_regularizer=l1(0.01)))
model.add(Dropout(0.01))  # 预防过拟合
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(256, 3, activation='relu', input_shape=input_shape, kernel_regularizer=l1(0.01)))
model.add(Dropout(0.01))  # 预防过拟合
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
######################################
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.01))  # 预防过拟合
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.01))  # 预防过拟合
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.01))  # 预防过拟合
model.add(Dense(num_classes, activation='softmax'))

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# import matplotlib.pyplot as plt
#
# # 输入数据
# images = ['img_1', 'img_2', 'img_3', 'img_4', 'img_5', 'img_6', 'img_7', 'img_8', 'img_9', 'img_10']
# vgg16_loss = [0.3552, 0.4561, 0.4425, 0.5269, 0.3301, 0.6224, 0.6141, 0.5874, 0.6393, 0.3468]
# voter_loss = [0.3545, 0.4697, 0.4539, 0.5461, 0.3672, 0.6609, 0.6427, 0.6232, 0.6754, 0.5746]
# color_recovery = [98.76, 98.85, 98.94, 99.06, 99.29, 99.52, 99.47, 99.54, 99.21, 98.77]
#
# # 创建图表
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 绘制柱状图
# x = range(len(images))
# bar_width = 0.3
#
# bar1 = ax.bar(x, vgg16_loss, width=bar_width, align='center', label='VGG16_loss')
# bar2 = ax.bar([i + bar_width for i in x], voter_loss, width=bar_width, align='center', label='Voter_loss')
# ax2 = ax.twinx()
# bar3 = ax2.plot(x, color_recovery, marker='o', color='red', label='Color recovery degree')
#
# # 设置图表属性
# ax.set_xlabel('Images')
# ax.set_ylabel('Loss')
# ax2.set_ylabel('Color recovery degree')
# ax.set_xticks([i + bar_width / 2 for i in x])
# ax.set_xticklabels(images)
# ax.legend(loc='upper left')
# ax2.legend(loc='upper right')
#
# # 添加数值标签
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
#
# autolabel(bar1)
# autolabel(bar2)
#
# # 显示图表
# plt.title('NoGAN Evaluation')
# plt.tight_layout()
# plt.show()
