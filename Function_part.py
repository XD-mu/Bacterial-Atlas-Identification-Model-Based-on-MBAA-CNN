# 标准库导入
import os
import sys
import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
import pickle
# 通用第三方库导入
import matplotlib as plt
import multiprocessing
from pylab import *
from scipy.signal import find_peaks
# 机器学习和数据处理库导入
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# TensorFlow 和 Keras 相关库导入
from tensorflow.keras import backend as K
from tensorflow.keras import Input, mixed_precision
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Attention, Lambda, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l1, l2, l1_l2
# 项目特定的导入
from Function_part import *
from pretty_confusion_matrix import pp_matrix_from_data

# 假设有一个全局字典来存储损失记录
loss_records = {}
# 全局变量来跟踪总进度
total_progress = multiprocessing.Value('i', 0)
total_epochs = multiprocessing.Value('i', 0)
# #学习率回归函数
# def lr_schedule(epoch, lr):
#     initial_lr = 0.01  # 初始学习率
#     decay_factor = 0.2  # 学习率衰减因子
#     decay_epochs = 50  # 学习率每隔几个epoch衰减一次
#     min_lr = 0.0001  # 最小学习率

#     if epoch > 0 and epoch % decay_epochs == 0:
#         new_lr = lr * decay_factor  # 学习率衰减
#         new_lr = max(new_lr, min_lr)  # 学习率不低于最小值
#         print(f'Learning rate decayed. New learning rate: {new_lr:.6f}')
#         return new_lr
#     else:
#         return lr
def cosine_lr_schedule(epoch, initial_lr, min_lr=0.00001, warmup_epochs=100, total_epochs=2000):
    if epoch < warmup_epochs:
        # Linearly increase learning rate for the first 'warmup_epochs' epochs
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay of the learning rate after warmup
        decayed = (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))) / 2
        lr = (initial_lr - min_lr) * decayed + min_lr
    return lr  
def load_pickle_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data
#均峰值外加局部的多层次空间中动态确定搜索窗口
def dynamic_peak_split_adjusted(input_data, target_ratio=0.33):
    # 检测峰值
    peaks, properties = find_peaks(input_data.flatten(), prominence=1e-2)
    if len(peaks) < 2:
        return input_data, np.array([]), np.array([])

    average_peak = np.mean(peaks)
    total_length = len(input_data)
    segment_length = int(total_length * target_ratio)
    
    # 动态确定搜索窗口
    start_index = max(0, int(average_peak - segment_length / 2))
    end_index = min(total_length, start_index + segment_length)

    if start_index + 3 * segment_length > total_length:
        start_index = total_length - 3 * segment_length
        end_index = start_index + segment_length
    
    # 根据计算出的索引切分数据
    input_1 = input_data[:start_index]
    input_2 = input_data[start_index:end_index]
    input_3 = input_data[end_index:end_index + segment_length]
    print(len(input_1), len(input_2), len(input_3))
    return input_1.shape, input_2.shape, input_3.shape
#多头注意力机制结合卷积神经网络模型定义部分
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, num_heads,**kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0  # Ensure that embed_size is divisible by num_heads

        self.projection_dim = embed_size // num_heads
        self.query_dense = tf.keras.layers.Dense(embed_size)
        self.key_dense = tf.keras.layers.Dense(embed_size)
        self.value_dense = tf.keras.layers.Dense(embed_size)
        self.combine_heads = tf.keras.layers.Dense(embed_size)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (-1, x.shape[1], self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_size': self.embed_size,
            'num_heads': self.num_heads,
        })
        return config
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # print("Shape before separate heads:", query.shape)
        query = self.separate_heads(query, batch_size)
        # print("Shape after separate heads:", query.shape)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        # print("Shape before reshape:", attention.shape)
        # concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_size))
        temp_shape = tf.shape(attention)
        seq_len = temp_shape[1]
        concat_attention = tf.reshape(attention, (batch_size, seq_len, self.embed_size))

        # print("Shape after reshape:", concat_attention.shape)
        output = self.combine_heads(concat_attention)
        return output

def attention_subnetwork(input_tensor):
    x = MultiHeadSelfAttention(embed_size=256, num_heads=8)(input_tensor)
    x = Conv1D(512, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    return x

def create_model(input_shape, num_classes):
    if os.path.exists('./model/best_model.h5'):
        # model = tf.keras.models.load_model('./model/best_model.h5')
        model = tf.keras.models.load_model('./model/best_model.h5', custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention})
        print("Load the previous model")
        return model
    
    else:
        input_tensor = Input(shape=input_shape)
    
        input_1 = Lambda(lambda x: x[:, :input_shape[0]//3, :])(input_tensor)
        input_2 = Lambda(lambda x: x[:, input_shape[0]//3:2*input_shape[0]//3, :])(input_tensor)
        input_3 = Lambda(lambda x: x[:, 2*input_shape[0]//3:, :])(input_tensor)
        x1 = attention_subnetwork(input_1)
        x2 = attention_subnetwork(input_2)
        x3 = attention_subnetwork(input_3)
        #整合三段注意力机制结果
        merged_output = Concatenate(axis=1)([x1, x2, x3])
        x = Conv1D(512, 3, activation='relu', kernel_regularizer=l1_l2(0.009, 0.005))(merged_output)
        
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(256, 3, activation='relu', kernel_regularizer=l1_l2(0.007, 0.005))(x)

        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(128, 3, activation='relu', kernel_regularizer=l1_l2(0.008, 0.005))(x)
        # x = Dropout(0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Flatten()(x)

        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.008, 0.005))(x)
        # x = Dropout(0.05)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.009, 0.005))(x)
        # x = Dropout(0.02)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.005, 0.01))(x)
        # x = Dropout(0.01)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=input_tensor, outputs=x)
        print("Load the new model")
        return model
    
def Formal_Training_Function(input_shape, num_classes, model_dir, learning_rate, X_train_full, y_train_full, batch_size, num_epochs, X_val, y_val, X_test, y_test, labels, label_dict,epoch):
    # 正式训练
    model = create_model(input_shape, num_classes)
    model.compile(optimizer=Adagrad(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # 设置回调函数,根据cosine_lr_schedule自动修改休息率，注释掉另一行是只调用一个学习率
    checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), monitor='val_loss', save_best_only=True, mode='min')
    lr_callback = LearningRateScheduler(lambda epoch: cosine_lr_schedule(epoch, initial_lr=learning_rate), verbose=1)
    # lr_callback = LearningRateScheduler(lambda epoch: learning_rate, verbose=1)

    # 模型训练
    history = model.fit(X_train_full, y_train_full, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[checkpoint, lr_callback, TrainingLogCallback()])

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
    cmap = "Blues"

    # 画图
    plot_metrics(history, labels, model, X_test, y_test, label_dict)
    pp_matrix_from_data(y_test, y_pred_classes, columns=labels, lw=accuracy, cmap=cmap)
    ROC_PR_Plot(y_test, y_pred, labels, num_classes)

    return model, history, loss, accuracy
# 需要确保所有被引用的自定义模块和函数都已经定义好，并且在函数调用前导入。
   
def plot_metrics(history, unique_labels,model,X_test,y_test,label_dict):
    # Set color scheme
    colors = ['#2c7bb6', '#d7191c', '#fdae61', '#AB8C7B', '#f46d43', '#d9ef8b', '#1a9641', '#c6dbef'][
             :len(unique_labels)]
    # Plot accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], color=colors[0], label='Train')
    plt.plot([x + 0.0 for x in history.history['val_accuracy']], color=colors[1], linestyle='--', label='Validation')

    plt.title('Train Accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig("./Results/Train Accuracy.svg", format="svg")
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
    plt.savefig("./Results/Train Loss.svg", format="svg")
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
    plt.show()

    # Plot recall
    plt.figure(figsize=(8, 6))
    plt.bar(unique_labels, recall, color=colors)
    plt.title('Model Recall', fontsize=16)
    plt.ylabel('Recall', fontsize=14)
    plt.xlabel('Label', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.show()

def ROC_PR_Plot(y_test, y_pred, labels,num_classes):
    num_classes = len(labels)

    # 画ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig("./Results/ROC曲线.svg", format="svg")
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
    plt.figure()
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-Recall curve of {0} (area = {1:0.2f})'
                 ''.format(labels[i], average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower right")
    plt.savefig("./Results/PR曲线.svg", format="svg")
    plt.show()
    
def load_data(data_dir,min_ndim):
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
    return np.array(X, dtype=object), np.array(y, dtype=object)

def load_and_process_data(origin_folder_path, train_dir, test_dir, min_ndim, max_length):
    # 获取细菌标签
    labels = []
    for root, dirs, files in os.walk(origin_folder_path):
        for dir in dirs:
            labels.append(dir)
    labels = [label for label in labels if label != '.ipynb_checkpoints']

    # 加载训练数据和测试数据
    X_train, y_train = load_data(train_dir, min_ndim)
    X_test, y_test = load_data(test_dir, min_ndim)

    # 将序列数据填充到相同的长度
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post', dtype='float32')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post', dtype='float32')

    # 将标签编码为整数
    unique_labels = np.unique(y_train)
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    y_train = np.array([label_dict[label] for label in y_train])
    y_test = np.array([label_dict[label] for label in y_test])
    num_classes = len(unique_labels)

    # 划分训练集和验证集
    X_train_full, y_train_full = X_train.copy(), y_train.copy()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train_next, X_val_next, y_train_next, y_val_next = train_test_split(X_val, y_val, test_size=0.08, random_state=42)
    indices_to_remove = []
    for i in range(len(X_val_next)):
        for j in range(len(X_train_full)):
            if np.array_equal(X_val_next[i], X_train_full[j]):
                indices_to_remove.append(j)
    X_train_full = np.delete(X_train_full, indices_to_remove, axis=0)
    y_train_full = np.delete(y_train_full, indices_to_remove, axis=0)

    # 修改训练数据形状
    X_train_full = np.array([arr[:, 1] for arr in X_train_full])
    X_train = np.array([arr[:, 1] for arr in X_train])
    X_val = np.array([arr[:, 1] for arr in X_val])
    X_test = np.array([arr[:, 1] for arr in X_test])

    X_train_full = X_train_full.reshape(X_train_full.shape[0], 1500, 1)
    X_train = X_train.reshape(X_train.shape[0], 1500, 1)
    X_val = X_val.reshape(X_val.shape[0], 1500, 1)
    X_test = X_test.reshape(X_test.shape[0], 1500, 1)
    # 转换为 TensorFlow 数据集
    def create_dataset(data, labels):
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.shuffle(len(data))
        return dataset

    X_train_full_ds = create_dataset(X_train_full, y_train_full)
    X_train_ds = create_dataset(X_train, y_train)
    X_val_ds = create_dataset(X_val, y_val)
    X_test_ds = create_dataset(X_test, y_test)

    X_train_full_ds = X_train_full_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    X_train_ds = X_train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    X_val_ds = X_val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    X_test_ds = X_test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return X_train_full, y_train_full, y_test, X_train, X_val, X_test, y_train, y_val, num_classes, label_dict, labels, unique_labels
#数据增强函数组
def add_noise(data, noise_level=0.001):
    noise = np.random.normal(loc=0, scale=noise_level, size=data.shape)
    return data + noise

def scale_data(data, scale_range=(0.999, 1.001)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale

def shift_data(data, shift_max=1):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(data, shift, axis=1)

def augment_data(data):
    rand = np.random.rand()
    if rand < 1/3:
        data = add_noise(data)
    elif 1/3 <= rand < 2/3:
        data = scale_data(data)
    else:
        data = shift_data(data)
    return data
#并行学习率
def generate_learning_rates():
    ranges = [(0.0001, 0.001), (0.001, 0.01), (0.01, 0.1)]
    learning_rates = []
    for r in ranges:
        learning_rates.extend(np.random.uniform(r[0], r[1], 5))
    return learning_rates

def train_model_with_lr(learning_rate, X_train, y_train, X_val, y_val, batch_size, num_epochs, input_shape, num_classes):
    model = create_model(input_shape, num_classes)
    model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    with total_epochs.get_lock():
        total_epochs.value += num_epochs

    callbacks = [TrainingLogCallback(), ProgressCallback(num_epochs)]
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=callbacks)
    loss_records[learning_rate] = history.history['loss']
    return history.history['loss'][-1]

def test_learning_rates():
    # 确保 loss_records 已经被填充
    if not loss_records:
        print("没有可用的训练记录。请先运行训练。")
        return
    
    plt.figure(figsize=(400, 200))
    for lr, losses in loss_records.items():
        plt.plot(losses, label=f'LR: {lr}')

    plt.title('Loss vs. Epochs for Different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.gca().xaxis.set_major_formatter(plt.FixedFormatter([round(i, 10) for i in plt.gca().get_xlim()]))
    plt.gca().yaxis.set_major_formatter(plt.FixedFormatter([round(i, 10) for i in plt.gca().get_ylim()]))

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()  

    plt.savefig("./Results/测试多种学习率测试对比图.svg", format="svg")

    # 显示图形
    plt.show()
    print("Draw Completed!!!Please Check!!!")

#日志记录
def save_training_log(epoch, loss, training_accuracy, model_dir, elapsed_time, learning_rate):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    loss_str = f"{loss:.4f}"
    training_accuracy_str = f"{training_accuracy:.4f}"
    lr_str = f"{learning_rate:.6f}"

    elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
    log_message = f"{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Training Accuracy] {training_accuracy_str} [Elapsed Time] {elapsed_time_str} [LR] {lr_str}\n"

    with open(os.path.join(model_dir, 'training_log.txt'), 'a') as log_file:
        log_file.write(log_message)

class TrainingLogCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = datetime.datetime.now()
        self.training_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        current_time = datetime.datetime.now()
        total_elapsed_time = (current_time - self.start_time).total_seconds()
        logs = logs or {}

        learning_rate = self.model.optimizer.lr.numpy()

        training_accuracy = logs.get('accuracy')

        total_elapsed_time_str = str(datetime.timedelta(seconds=int(total_elapsed_time)))

        self.training_accuracies.append(training_accuracy)

        save_training_log(epoch, logs.get('loss'), training_accuracy, './model', total_elapsed_time, learning_rate)


    def get_training_accuracies(self):
        return self.training_accuracies        

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs_per_run):
        super().__init__()
        self.local_epochs = total_epochs_per_run

    def on_epoch_end(self, epoch, logs=None):
        with total_progress.get_lock():
            total_progress.value += 1
            total_percentage = (total_progress.value / total_epochs.value) * 100
            sys.stdout.write(f"\rTotal Training Progress: {total_percentage:.2f}%")
            sys.stdout.flush()