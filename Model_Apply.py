import os
import numpy as np
import matplotlib
from keras.models import load_model
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import precision_recall_curve, average_precision_score
#字体路径根据你的系统来调整
# font = FontProperties(fname='./font/songti.ttf', size=12)
# plt.rcParams['font.sans-serif'] = [font.get_name()]
# plt.rcParams['axes.unicode_minus'] = False

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

matplotlib.rcParams['axes.unicode_minus'] = False
#支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 设置文件夹目录
train_dir = './Final_Data_Ori/train'  # 训练数据文件夹
test_dir = './Final_Data_Ori/test'  # 测试数据文件夹
model_dir = './model'  # 模型保存文件夹
batch_size=1
#自动获取所有的细菌标签
origin_folder_path='./Origin_Data/data2'
labels=[]


# 加载测试数据函数
def load_data(data_dir):
    X = []
    y = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            data = np.loadtxt(file_path)
            if data.ndim < 2:
                data = np.expand_dims(data, axis=0)
            X.append(data)
            label = filename.split("_")[0]
            y.append(label)
    return np.array(X), np.array(y)

# 遍历标签目录
for root, dirs, files in os.walk(origin_folder_path):
    for dir in dirs:
        # 将文件夹名称添加到标签列表中
        labels.append(dir)
labels = [label for label in labels if label != '.ipynb_checkpoints']
print(labels)

X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)
#############################
# 将标签编码为整数
unique_labels = np.unique(y_train)
label_dict = {label: i for i, label in enumerate(unique_labels)}
y_train = np.array([label_dict[label] for label in y_train])
y_test = np.array([label_dict[label] for label in y_test])
num_classes = len(unique_labels)
# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 加载模型85%model.h5
# best_model_path = os.path.join(model_dir, 'best_model.h5')
best_model_path = os.path.join(model_dir, '96%.h5')
model = load_model(best_model_path)

# 进行预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)

# # 展示预测结果
# print('---------------------------------------------')
# print('结果标签:', y_pred_classes)
# cout = 0
# print('------------------预测结果展示-------------------------')
# for file_name in os.listdir(test_dir):
#     predicted_label = labels[int(y_pred_classes[cout])]
#     file_name = file_name.split('_')[0]
#     print('True Value:{}            Predict Value : {}'.format(file_name, predicted_label))
#     cout += 1
# print('---------------------------------------------')
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# print(y_test)
# print(y_pred_classes)
cmap = "PuRd"
pp_matrix_from_data(y_test, y_pred_classes,columns=labels,lw=accuracy,cmap=cmap)
print(y_test)
print('----------------------------------------')
print(y_pred_classes)
print('----------------------------------------')
# print(labels)
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
plt.show()