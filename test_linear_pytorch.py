import gzip
import sys
import os
import time
import struct 
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
print("matplotlib backend:",matplotlib.get_backend())

files = ['t10k-images-idx3-ubyte.gz',
         't10k-labels-idx1-ubyte.gz',
         'train-images-idx3-ubyte.gz',
         'train-labels-idx1-ubyte.gz']
directory = "../MNIST/"
filepaths = [directory + i for i in files]
for i in filepaths:
    print(i)
def readlabels(path):
    with gzip.open(path, 'rb') as f1: # 以二进制方式读取
        data = f1.read()
        return [int(i) for i in data[8:]] # 舍弃前8个字节
test_labels = np.array(readlabels(filepaths[1]))
train_labels = np.array(readlabels(filepaths[3]))
def readimgs(path):
    with gzip.open(path, 'rb') as f1:
        data = f1.read()
        n_imgs = (len(data)-16)//(28*28)  # 舍弃前16个字符
        fmt = '>{}B'.format(28*28)        # > 表示大端模式， 读取28*28个8位像素
        images = np.empty((n_imgs,28,28))
        for i in range(n_imgs):
            offset = 16+28*28*i
            images[i] = np.array(struct.unpack_from(fmt, data, offset)).reshape((28,28))
    return images
test_imgs = readimgs(filepaths[0])
train_imgs = readimgs(filepaths[2])


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28,40)
        self.fc2 = nn.Linear(40,30)
        self.fc3 = nn.Linear(30,10)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x),0.1)
        x = F.leaky_relu(self.fc2(x),0.1)
        x = self.fc3(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
batch_size = 10

test_labels = torch.from_numpy(test_labels).long()
train_labels = torch.from_numpy(train_labels).long()
test_imgs = torch.from_numpy(test_imgs).view(10000,-1).float()
train_imgs = torch.from_numpy(train_imgs).view(60000,-1).float()

t_start = time.time()
for epoch in range(10000):  # loop over the dataset multiple times
    idxes = torch.randint(60000, size=(1,batch_size)).view(-1)
    x = train_imgs[idxes]
    y = train_labels[idxes]
    
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch%100==99:
        print("iteration:",epoch,"\tloss:",loss.item())

print("training time:",time.time()-t_start)
test_size = 9000
idxes = np.random.randint(10000, size=test_size)
x = test_imgs[idxes]
y = test_labels[idxes]
accuracy = torch.sum(torch.argmax(net(x),axis=-1) == y).item()  / test_size
print("test accuracy:",accuracy)

