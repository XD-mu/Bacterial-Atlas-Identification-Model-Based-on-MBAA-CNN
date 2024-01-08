import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pylab import *
import matplotlib
import tensorflow as tf
from keras.layers import *
from tensorflow.keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import load_model, Model, Sequential
from tensorflow.keras import backend as K
from keras.regularizers import l1, l2
from pretty_confusion_matrix import pp_matrix_from_data
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Attention, Lambda, Concatenate
from tensorflow.keras import Input
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import precision_recall_curve, average_precision_score
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import l1_l2
from Function_part import MultiHeadSelfAttention
from Function_part import *
#字体路径根据你的系统来调整

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 设置文件夹目录
train_dir = './Final_Data/data3/train'  # 训练数据文件夹
test_dir = './Final_Data/data3/test'  # 测试数据文件夹
model_dir = './model'  # 模型保存文件夹
batch_size=8
min_ndim=2
max_length=1500
#######################################
# 自动获取所有的细菌标签
origin_folder_path = './Origin_Data/data3/all'
labels = []

# 获取细菌标签
for root, dirs, files in os.walk(origin_folder_path):
    for dir in dirs:
        # 将文件夹名称添加到标签列表中
        labels.append(dir)
labels = [label for label in labels if label != '.ipynb_checkpoints']
print(labels)

# 加载训练数据和测试数据
X_train, y_train = load_data(train_dir,min_ndim)
X_test, y_test = load_data(test_dir,min_ndim)

# 将序列数据填充到相同的长度
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', dtype='float32')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', dtype='float32')
#############################
# 将标签编码为整数
unique_labels = np.unique(y_train)
label_dict = {label: i for i, label in enumerate(unique_labels)}
y_train = np.array([label_dict[label] for label in y_train])
y_test = np.array([label_dict[label] for label in y_test])
num_classes = len(unique_labels)

# 划分训练集和验证集
X_train_full, y_train_full = X_train.copy(), y_train.copy()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train_next, X_val_next, y_train_next, y_val_next = train_test_split(X_val, y_val, test_size=0.07, random_state=42)
indices_to_remove = []
for i in range(len(X_val_next)):
    for j in range(len(X_train_full)):
        if np.array_equal(X_val_next[i], X_train_full[j]):
            indices_to_remove.append(j)
X_train_full = np.delete(X_train_full, indices_to_remove, axis=0)
y_train_full = np.delete(y_train_full, indices_to_remove, axis=0)
# print(X_train_full)
# print("X_val_next")
# print(X_val_next)
# print("indices_to_remove")
# print(indices_to_remove)
#修改训练数据形状（为了保证输入数据的有序性，我们每份细菌数据只取第二列光谱数据，修改后维度为[1500,1]）
X_train_full = np.array([arr[:, 1] for arr in X_train_full])
X_train = np.array([arr[:, 1] for arr in X_train])
X_val = np.array([arr[:, 1] for arr in X_val])
X_test = np.array([arr[:, 1] for arr in X_test])

X_train_full = X_train_full.reshape(X_train_full.shape[0], 1500, 1)
X_train = X_train.reshape(X_train.shape[0], 1500, 1)
X_val = X_val.reshape(X_val.shape[0],1500,1)
X_test = X_test.reshape(X_test.shape[0],1500,1)



# 加载模型
best_model_path = os.path.join(model_dir, '98.63%.h5')
model = load_model(best_model_path,custom_objects={'MultiHeadSelfAttention':MultiHeadSelfAttention})

# 进行预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)

# # 展示预测结果
# print('---------------------------------------------')
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

cmap = "Blues"
pp_matrix_from_data(y_test, y_pred_classes,columns=labels,lw=accuracy,cmap=cmap)
print(y_test)
print('----------------------------------------')
print(y_pred_classes)
print('----------------------------------------')
##########ROC曲线##############
# 画ROC曲线
# 分别绘制每个类别的ROC曲线
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test==i, y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 加颜色和标签
colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(labels[i], roc_auc[i]))

# 添加一些ROC指令
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.savefig("./模型部署结果/ROC曲线.svg", format="svg")
plt.show()

# 计算每个类别的平均精度分数
average_precision = dict()
for i in range(num_classes):
    average_precision[i] = average_precision_score(y_test == i, y_pred[:, i])

# 计算每个类别的精度-召回率曲线
precision = dict()
recall = dict()
for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test == i, y_pred[:, i])

# 绘制每个类别的精度-召回率曲线
colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
for i, color in zip(range(num_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='Precision-Recall curve of {0} (area = {1:0.2f})'
             ''.format(labels[i], average_precision[i]))

# 添加一些PR曲线指令
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="lower right")
plt.savefig("./模型部署结果/PR曲线.svg", format="svg")
plt.show()