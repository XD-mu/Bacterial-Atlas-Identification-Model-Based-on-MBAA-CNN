import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adagrad
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.utils import compute_sample_weight
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
import glob
import torch.optim.lr_scheduler as lr_scheduler

font = FontProperties(fname='./font/songti.ttf')  # 字体的路径，这个路径需要根据你的系统来调整
train_dir = './Final_Data/train'  # 训练数据文件夹
test_dir = './Final_Data/test'  # 测试数据文件夹
model_dir = './model'  # 模型保存文件夹
origin_folder_path='./Origin_Data'
labels=[]
for root, dirs, files in os.walk(origin_folder_path):
    for dir in dirs:
        labels.append(dir)

num_epochs = 1000
batch_size = 32
min_ndim = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create a custom dataloader
class MyDataset(Dataset):
    def __init__(self, data_dir, label_dict):
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.files = glob.glob(data_dir + '/*.txt')
        self.le = LabelEncoder().fit(labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.loadtxt(self.files[idx])
        if data.ndim < min_ndim:
            data = np.expand_dims(data, axis=0)
        label = self.le.transform([os.path.basename(self.files[idx]).split("_")[0]])[0]
        return torch.Tensor(data), torch.Tensor([label])

# define model architecture
class Net(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1201, 256, 2)
        self.drop1 = nn.Dropout(0.05)
        self.pool1 = nn.MaxPool1d(1)
        self.lstm = nn.LSTM(1, 100, batch_first=True)
        self.drop2 = nn.Dropout(0.05)
        self.fc1 = nn.Linear(100, 128)
        self.drop3 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x, _ = self.lstm(x)
        x = self.drop2(x)
        x = x.contiguous().view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x

le = LabelEncoder().fit(labels)
train_data = MyDataset(train_dir, le)
test_data = MyDataset(test_dir, le)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
input_shape = next(iter(train_loader))[0].shape
num_classes = len(labels)
model = Net(input_shape, num_classes).to(device)
optimizer = Adagrad(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def lr_schedule(optimizer, epoch):
    if epoch > 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    return optimizer

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).long()
            output = model(data)
            test_loss += criterion(output, target.view(-1)).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# training loop
for epoch in range(1, num_epochs + 1):
    optimizer = lr_schedule(optimizer, epoch)
    train(epoch)
    test(epoch)

# saving model
torch.save(model.state_dict(), './model/best_model.pt')
