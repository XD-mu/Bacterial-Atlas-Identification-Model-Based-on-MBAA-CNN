import os
import datetime
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

import matplotlib as plt
import multiprocessing
from pylab import *
from scipy.signal import find_peaks

import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from tensorflow.keras import backend as K
from tensorflow.keras import Input, mixed_precision
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Attention, Lambda, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l1, l2, l1_l2

from Function_part import *
from pretty_confusion_matrix import pp_matrix_from_data

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
loss_records = {}

# 设置文件夹目录
origin_folder_path = './Origin_Data'
train_dir = './Final_Data/train'  # 训练数据文件夹
test_dir = './Final_Data/test'  # 测试数据文件夹
model_dir = './model'  # 模型保存文件夹
################################################
# 超参数设置
TEST_MULTIPLE_LEARNING_RATES = False  # 将这个值设置为True以测试多个学习率
learning_rate = 0.01  # 如果不测试多个学习率，使用这个确定的学习率
batch_size = 8
min_ndim = 2
num_epochs = 2000
max_length = 1500
#######################################
X_train_full, y_train_full, y_test, X_train, X_val, X_test, y_train, y_val, num_classes, label_dict, labels, unique_labels = load_and_process_data(origin_folder_path, train_dir, test_dir, min_ndim, max_length)

input_shape = X_train[0].shape
print(f'输入维度:{input_shape}' )

if TEST_MULTIPLE_LEARNING_RATES:
    total_progress.value = 0
    total_epochs.value = 0
    learning_rates = generate_learning_rates()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(train_model_with_lr, lr, X_train_full,y_train_full, X_val, y_val, batch_size, num_epochs, input_shape, num_classes) for lr in learning_rates]
        losses = [future.result() for future in futures]
    best_lr_index = np.argmin(losses)
    best_lr = learning_rates[best_lr_index]
    print(f"直接筛选得到的Best Learning Rate1: {best_lr}")
    test_learning_rates()
else:  
    model, history, test_loss, test_accuracy = Formal_Training_Function(input_shape, num_classes, model_dir, learning_rate, X_train_full, y_train_full, batch_size, num_epochs, X_val, y_val, X_test, y_test, labels, label_dict)

