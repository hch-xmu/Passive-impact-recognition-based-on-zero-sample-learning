import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from torch import device
from torch.optim import optimizer
from torch.utils.data import DataLoader, Dataset
from util import DSPCA

from read_data import create_data
from scipy.io import loadmat


#
class my_dataset(Dataset):
    def __init__(self, data, attribute_label):
        super(my_dataset, self).__init__()
        self.data = data
        self.attribute_label = attribute_label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        batch_data = self.data[index]
        batch_label = self.attribute_label[index]
        return batch_data, batch_label


device = torch.device('cuda')

np.random.seed(904)
# 第一步是特征提取。采用监督主成分分析提取属性相关特征。将从原始的 52 个变量中提取 20 个要素。


def feature_extraction(traindata, testdata, train_attributelabel, test_attributelabel):
    trainfeatures = []
    testfeatures = []
    for i in range(train_attributelabel.shape[1]):
        spca = DSPCA(10)
        spca.fit(traindata, train_attributelabel[:, i])
        trainfeatures.append(spca.transform(traindata))
        testfeatures.append(spca.transform(testdata))
        # trainfeatures=np.column_stack(trainfeatures);
        # testfeatures=np.column_stack(testfeatures);
    return np.column_stack(trainfeatures), np.column_stack(testfeatures)
# FDAT模型


def pre_model(model, traindata, train_attributelabel, testdata, testlabel, attribute_matrix):
    model_dict = {'rf': RandomForestClassifier(n_estimators=100), 'NB': GaussianNB(
    ), 'SVC_linear': SVC(kernel='linear'), 'LinearSVC': LinearSVC()}
    model = 'NB'
    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata, train_attributelabel[:, i])
            res = clf.predict(testdata)
        else:
            res = np.zeros(testdata.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T

    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = (np.sum(np.square(attribute_matrix - pre_res), axis=1)).argmin()
        label_lis.append(np.unique(testlabel)[loc])
    label_lis = np.mat(np.row_stack(label_lis))
    return test_pre_attribute, label_lis, testlabel


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Embedding_Net(nn.Module):

    def __init__(self, dim, lambda_):
        super(Embedding_Net, self).__init__()
        self.l11 = nn.Linear(310, dim[0])
        self.l12 = nn.Linear(dim[0], dim[0])
        self.l13 = nn.Linear(dim[0], dim[0])
        self.l14 = nn.Linear(dim[0], dim[0])
        self.l15 = nn.Linear(dim[0], dim[0])
        self.l16 = nn.Linear(dim[0], dim[0])
        self.l17 = nn.Linear(dim[0], dim[0])
        self.l18 = nn.Linear(dim[0], dim[1])

        self.l19 = nn.Linear(2*dim[1], 310)

        self.l21 = nn.Linear(31, dim[0])
        self.l22 = nn.Linear(dim[0], dim[1])
        self.l23 = nn.Linear(2*dim[1], 31)

        self.bn1 = nn.BatchNorm1d(dim[0])
        self.bn2 = nn.BatchNorm1d(dim[1])

        self.lambda_ = lambda_

    def compability_loss(self, z1, z2):
        N, D = z1.shape

        c = self.bn2(z1).T @ self.bn2(z2)/N

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag+self.lambda_[3]*off_diag

        return loss

    def compute_loss(self, z1, z2, x, a, x_, a_):
        loss_R1 = self.lambda_[0]*F.mse_loss(a, a_)
        loss_R2 = self.lambda_[1]*F.mse_loss(x, x_)
        loss_CM = self.compability_loss(z1, z2)
        loss_CM = self.lambda_[2]*loss_CM
        loss = loss_R1+loss_R2+loss_CM
        return loss_R1, loss_R2, loss_CM, loss

    def transform(self, x, a):
        z1 = self.l11(x)
        z1 = self.l12(z1)
        z1 = self.l13(z1)
        z1 = self.l14(z1)
        z1 = self.l15(z1)
        z1 = self.l16(z1)
        z1 = self.l17(z1)
        z1 = self.l18(z1)

        z2 = self.l21(a)
        z2 = torch.relu(self.bn1(z2))
        z2 = self.l22(z2)
        return z1, z2

    def reconstruction(self, z1, z2):
        f1 = torch.cat([z1, z2], dim=1)
        f2 = torch.cat([z2, z1], dim=1)
        x_ = self.l19(f1)
        a_ = torch.sigmoid(self.l23(f2))
        return x_, a_

    def forward(self, x, a):
        z1, z2 = self.transform(x, a)
        x_, a_ = self.reconstruction(z1, z2)

        loss_R1, loss_R2, loss_CM, loss = self.compute_loss(
            z1, z2, x, a, x_, a_)
        package = {'z1': z1, 'z2': z2, 'x': x, 'x_': x_, 'r1': loss_R1,
                   'r2': loss_R2, 'cm': loss_CM, 'loss': loss}

        return package


modes = ['rf']  # 'rf'
# 导入橡胶球的数据，每个区域4*20
data_xj = loadmat('E:/被动冲击/数据集/XJ/data.mat')
data_xj = data_xj['dataset']

# 导入铁球的数据，每个区域4*10
data_tg = loadmat('E:/被动冲击/数据集/TG/matlab.mat')
data_tg = data_tg['dataset']

# 导入属性矩阵
attribute_matrix_ = pd.read_excel('E:/被动冲击/数据集/attribute06.xlsx')
attribute_matrix = attribute_matrix_.values
# 训练集和测试集的所有标签
train_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
test_index = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
train_index.sort()
test_index.sort()

# 由于数据集中包含数据和标签，对其进行分离
# train集合（橡胶球）
# 初始化
trainlabel = []
train_attributelabel = []
traindata = []
# 开始赋值
traindata = data_xj[0:6400, :]
traindata = traindata.T
# 进行信号截取
traindata = traindata[:, 1500:1900]
# 热力图显示
image01 = traindata[0:4, :]
plt.figsize = (2, 5)
plt.imshow(image01, cmap='hot', aspect='auto')
plt.colorbar()  # 显示颜色标尺
plt.show()
plt.savefig("Second image.png")
# 将数据转换为热力图格式，（1280/4）*（4*400）

traindata_hot_1d = traindata.reshape([320, 1600])

for item in train_index:
    trainlabel += [item] * 20
    train_attributelabel += [attribute_matrix[item, :]]*20
# 将list转化为array
train_attributelabel = np.row_stack(train_attributelabel)
trainlabel = np.row_stack(trainlabel)

# test集合(铁球)
# 初始化
testlabel = []
test_attributelabel = []
testdata = []
# 开始赋值
testdata = data_tg[0:6400, :]
testdata = testdata.T
# 进行信号截取
testdata = testdata[:, 1500:1900]
# 将数据转换为热力图格式，（640/4）*（4*400）

testdata_hot_1d = testdata.reshape([160, 1600])

for item in test_index:
    test_attributelabel += [attribute_matrix[item-16, :]]*10
    testlabel += [item] * 10
# 将list转化为array
test_attributelabel = np.row_stack(test_attributelabel)
testlabel = np.row_stack(testlabel)

test_attributematrix = attribute_matrix_.iloc[test_index, :]
train_attributematrix = attribute_matrix_.iloc[train_index, :]

print("SPCA extracting feature (takes lots of time)...")
traindata, testdata = feature_extraction(
    traindata_hot_1d, testdata_hot_1d, train_attributelabel, test_attributelabel)


# 使用FDAT得到的精度值
# NB
_, y_pre, y_true = pre_model(
    'NB', traindata, train_attributelabel, testdata, testlabel, attribute_matrix)
original_acc_NB = accuracy_score(y_pre, y_true)
print('FDAT(NB)的acc是', original_acc_NB)
# rb
_, y_pre, y_true = pre_model(
    'rb', traindata, train_attributelabel, testdata, testlabel, attribute_matrix)
original_acc_rb = accuracy_score(y_pre, y_true)
print('FDAT(rb)的acc是', original_acc_rb)
# 将数据格式转换为torch格式
traindata = torch.from_numpy(traindata).float().to(device)
label = torch.from_numpy(trainlabel.squeeze()).long().to(device)
testdata = torch.from_numpy(testdata).float().to(device)

batch_size = 16
trainset = my_dataset(traindata, torch.from_numpy(
    train_attributelabel).float().to(device))
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

lambda_ = [1, 1e-3, 1, 0.25]
dim = [256, 64]
model = Embedding_Net(dim, lambda_=lambda_)
model.to(device)

#optimizer = optim.RMSprop(model.parameters(), lr=1e-2)
optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.99))

L1, L2, L3, L = [], [], [], []
model.train()

accs = []
acc_log = []
best_acc = 0
for epoch in range(20000):
    model.train()
    for batch, (batch_data, batch_label) in enumerate(train_loader):

        optimizer.zero_grad()
        package = model(batch_data, batch_label)
        loss_R1, loss_R2, loss_CM, loss = package['r1'], package['r2'], package['cm'], package['loss']

        loss.backward()
        optimizer.step()

        L1.append(loss_R1.item())
        L2.append(loss_R2.item())
        L3.append(loss_CM.item())
        L.append(loss.item())

    model.eval()
    with torch.no_grad():
        train_package = model(traindata, torch.from_numpy(
            train_attributelabel).float().to(device))
        f_train = train_package['z1']
        f_train = torch.cat([f_train, traindata], dim=1).detach().cpu().numpy()

        test_package = model(testdata, torch.from_numpy(
            test_attributelabel).float().to(device))
        f_test = test_package['z1']
        f_test = torch.cat([f_test, testdata], dim=1).detach().cpu().numpy()

        test_preattribute, label_lis, testlabel = pre_model(
            modes[0], f_train, train_attributelabel, f_test, testlabel, attribute_matrix)
        acc = accuracy_score(label_lis, testlabel)
        accs.append(acc)
        if acc > best_acc:
            best_acc = acc
    acc_log.append([epoch, best_acc])
    print('epoch:{:d}, best_acc:{:.4f}'.format(epoch, best_acc))

print('finished! FDAT:NB:{:.4f} rb:{:.4f}, SCE:{:.4f}'.format(
    original_acc_NB, original_acc_rb, best_acc))
# 画出acc图像
acclog = np.row_stack(acc_log)
plt.plot(acclog[:, 0], acclog[:, 1])
plt.xlabel('epoch')
plt.ylabel('accuracy')
