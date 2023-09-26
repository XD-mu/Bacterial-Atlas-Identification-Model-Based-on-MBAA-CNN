import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from pylab import *
import random
import matplotlib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from keras.regularizers import l1, l2
from keras.layers import BatchNormalization
from keras.layers import *
from tensorflow.keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import load_model, Model, Sequential
from tensorflow.keras import backend as K
from keras.regularizers import l1, l2
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.utils import compute_sample_weight
from tensorflow.keras.utils import to_categorical
from keras.activations import softmax
from sklearn.metrics import precision_recall_curve, average_precision_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

matplotlib.rcParams['axes.unicode_minus'] = False
# 支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 设置文件夹目录
train_dir = './Final_Data/data2/train'  # 训练数据文件夹
test_dir = './Final_Data/data2/test'  # 测试数据文件夹
model_dir = './model'  # 模型保存文件夹

# 自动获取所有的细菌标签
origin_folder_path = './Origin_Data/data2'
labels = []

# 自定义一个学习率回调函数
def lr_schedule(epoch, lr):
    initial_lr = 0.005  # 初始学习率
    decay_factor = 0.1  # 学习率衰减因子
    decay_epochs = 20  # 学习率每隔几个epoch衰减一次
    min_lr = 0.0001  # 最小学习率

    if epoch > 0 and epoch % decay_epochs == 0:
        new_lr = lr * decay_factor  # 学习率衰减
        new_lr = max(new_lr, min_lr)  # 学习率不低于最小值
        print(f'Learning rate decayed. New learning rate: {new_lr:.6f}')
        return new_lr
    else:
        return lr

# 保存照片
def save_plot(directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))


################################################
# 超参数设置
batch_size = 4
min_ndim = 2
num_epochs = 60


#################################################
# 定义卷积神经网络模型函数,l1(0.01)和l2(0.01)分别表示L1和L2正则化项的权重，可以根据需要调整。这样定义的模型在训练过程中，损失函数会自动加入正则化项，从而惩罚模型的复杂度。
def create_model(input_shape):
    # if any([os.path.isfile(os.path.join(model_dir, item)) for item in os.listdir(model_dir)]):
    try:
        model = load_model('./model/best_model.h5')
        return model
    except Exception as e:
        model = Sequential()

        #         # 第一层卷积
        #         model.add(Conv1D(256, 3, activation='relu', input_shape=input_shape, kernel_regularizer=l1(0.01)))
        #         model.add(BatchNormalization())
        #         model.add(Dropout(0.01))  # 更高的Dropout率
        #         model.add(MaxPooling1D(pool_size=2))

        #         # 第二层卷积
        #         model.add(Conv1D(128, 3, activation='relu', kernel_regularizer=l1(0.01)))
        #         model.add(BatchNormalization())
        #         model.add(Dropout(0.01))  # 更高的Dropout率
        #         model.add(MaxPooling1D(pool_size=2))

        #         model.add(Flatten())

        #         # 第一层全连接
        #         model.add(Dense(256, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        #         model.add(BatchNormalization())
        #         model.add(Dropout(0.05))  # 更高的Dropout率

        #         # 第二层全连接
        #         model.add(Dense(128, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        #         model.add(BatchNormalization())
        #         model.add(Dropout(0.05))  # 更高的Dropout率

        #         model.add(Dense(num_classes, activation='softmax'))

        #############################################
        model = Sequential()
        model.add(Conv1D(512, 3, activation='relu', input_shape=input_shape, kernel_regularizer=l1(0.01)))
        model.add(Dropout(0.1))  # 预防过拟合
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(256, 3, activation='relu', input_shape=input_shape, kernel_regularizer=l1(0.02)))
        model.add(Dropout(0.1))  # 预防过拟合
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(256, 3, activation='relu', input_shape=input_shape, kernel_regularizer=l1(0.01)))
        model.add(Dropout(0.1))  # 预防过拟合
        model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())
        ######################################
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Dropout(0.15))  # 预防过拟合
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.02), bias_regularizer=l2(0.01)))
        model.add(Dropout(0.1))  # 预防过拟合
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Dropout(0.1))  # 预防过拟合
        model.add(Dense(num_classes, activation='softmax'))
        return model


# 加载训练数据函数
def load_data(data_dir):
    X = []
    y = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            data = np.loadtxt(file_path)
            if data.ndim < min_ndim:
                data = np.expand_dims(data, axis=0)
            X.append(data)
            label = filename.split("_")[0]
            y.append(label)
    return np.array(X), np.array(y)


def plot_metrics(history, unique_labels):
    # Set color scheme
    colors = ['#2c7bb6', '#d7191c', '#fdae61', '#AB8C7B', '#f46d43', '#d9ef8b', '#1a9641', '#c6dbef'][
             :len(unique_labels)]
    # Plot accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], color=colors[0], label='Train')
    plt.plot([x + 0.0 for x in history.history['val_accuracy']], color=colors[1], linestyle='--', label='Validation')

    plt.title('Model Accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    save_plot('img', 'accuracy.png')
    plt.show()

    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], color=colors[0], label='train_loss')
    plt.plot(history.history['val_loss'], color=colors[1], linestyle='--', label='val_loss')
    plt.title('Model Loss', fontsize=16)
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    save_plot('img', 'loss.png')
    plt.show()

    # Calculate precision and recall for each class
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    precision = []
    recall = []
    for label in unique_labels:
        indices = np.where(y_test == label_dict[label])[0]
        tp = np.sum(y_pred[indices] == label_dict[label])
        fp = np.sum(y_pred[indices] != label_dict[label])
        fn = np.sum(y_pred != label_dict[label]) - tp
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))

    # Plot precision
    plt.figure(figsize=(8, 6))
    plt.bar(unique_labels, precision, color=colors)
    plt.title('Model Precision', fontsize=16)
    plt.ylabel('Precision', fontsize=14)
    plt.xlabel('Label', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    save_plot('img', 'precision.png')  # 保存图像
    plt.show()

    # Plot recall
    plt.figure(figsize=(8, 6))
    plt.bar(unique_labels, recall, color=colors)
    plt.title('Model Recall', fontsize=16)
    plt.ylabel('Recall', fontsize=14)
    plt.xlabel('Label', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    save_plot('img', 'recall.png')  # 保存图像
    plt.show()


# 获取细菌标签
for root, dirs, files in os.walk(origin_folder_path):
    for dir in dirs:
        # 将文件夹名称添加到标签列表中
        labels.append(dir)
labels = [label for label in labels if label != '.ipynb_checkpoints']
print(labels)

# 加载训练数据和测试数据
X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)
# X_train = tf.convert_to_tensor(X_trains, dtype=tf.float32)
# X_test = tf.convert_to_tensor(X_tests, dtype=tf.float32)
# y_test = tf.convert_to_tensor(y_tests, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_trains, dtype=tf.float32)
#############################
# 将标签编码为整数
unique_labels = np.unique(y_train)
label_dict = {label: i for i, label in enumerate(unique_labels)}
y_train = np.array([label_dict[label] for label in y_train])
y_test = np.array([label_dict[label] for label in y_test])
num_classes = len(unique_labels)
# print(y_train)
# print(y_test)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 创建模型
input_shape = X_train[0].shape
print('input_shape:', input_shape)
model = create_model(input_shape)
# 设置优化器，学习率随着训练轮次变化
optimizer = Adagrad(learning_rate=K.get_value(lr_schedule(num_epochs, 0.005)))
# 编译模型
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 设置回调函数，保存在每个epoch中的最佳模型
checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), monitor='val_loss', save_best_only=True,
                             mode='min')
# 模型训练
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val),
                    callbacks=[checkpoint])
# 引入画图函数
print(labels)

plot_metrics(history, labels)
# 加载最佳模型
best_model_path = os.path.join(model_dir, 'best_model.h5')
model.load_weights(best_model_path)
# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)

# 进行预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print('---------------------------------------------')
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

cmap = "PuRd"
pp_matrix_from_data(y_test, y_pred_classes, columns=labels, lw=accuracy, cmap=cmap)


# 将预测结果划分为17个区间段并计算归一化分数
num_intervals = 17
intervals = np.linspace(299, 2000, num_intervals + 1)
interval_scores = np.zeros(num_intervals)

for i in range(num_intervals):
    lower_bound = intervals[i]
    upper_bound = intervals[i + 1]
    indices = np.where((lower_bound <= y_test) & (y_test <= upper_bound))[0]

    if len(indices) == 0:
        interval_scores[i] = 0.0
    else:
        correct_predictions = np.sum(y_pred_classes[indices] == y_test[indices])
        interval_accuracy = correct_predictions / len(indices)
        interval_scores[i] = interval_accuracy

# 归一化分数
min_score = np.min(interval_scores)
max_score = np.max(interval_scores)
normalized_scores = (interval_scores - min_score) / (max_score - min_score)

# 输出每个区间段的归一化分数
for i in range(num_intervals):
    lower_bound = intervals[i]
    upper_bound = intervals[i + 1]
    print(f"Interval [{lower_bound}-{upper_bound}]: Normalized Score = {normalized_scores[i]:.4f}")
