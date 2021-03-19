import gzip
import sys
import os
import time
import struct 
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt



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
test_imgs = readimgs(filepaths[0]).reshape((10000,-1))
train_imgs = readimgs(filepaths[2]).reshape((60000,-1))



import mytorch as nn

class myMNIST(nn.Graph):
    def __init__(self):
        super().__init__()
        self.input = nn.Variable()
        self.linear1 = nn.linear(28*28,40)
        self.relu1 = nn.leaky_relu(0.1)
        self.linear2 = nn.linear(40,30)
        self.relu2 = nn.leaky_relu(0.1)
        self.linear3 = nn.linear(30,10)
        self.loss = nn.loss_softmax_cross_entropy()
        
    def forward(self,x,y):
        self.input.connect(x)
        self.linear1.connect(self.input)
        self.relu1.connect(self.linear1)
        self.linear2.connect(self.relu1)
        self.relu2.connect(self.linear2)
        self.linear3.connect(self.relu2)
        self.loss.connect(self.linear3, y)
        self.outputs = np.argmax(self.loss.outputs,axis=-1)


model = myMNIST()
batch_size = 10
epoch = 10000
min_step = 0.001
max_step = 0.01

t_start = time.time()
for i in range(epoch):
    
    idxes = np.random.randint(60000, size=batch_size)
    x = train_imgs[idxes]
    y = train_labels[idxes]
    
    model.forward(x,y)
    model.backward()
    model.step(max_step-(max_step-min_step)*i/epoch)
    if np.isnan(model.loss.loss):
        break
    if i % 100 == 99:
        print("iteration =",i,"\t\tloss =",model.loss.loss, "\t\tbatch_accuracy =",np.sum(model.outputs==y)/batch_size)

print("training time:",time.time()-t_start)
test_size = 9000
idxes = np.random.randint(10000, size=test_size)
x = test_imgs[idxes]
y = test_labels[idxes]
model.forward(x,y)

accuracy = np.sum(model.outputs == y) / test_size
print("test accuracy:",accuracy)
