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
test_imgs = readimgs(filepaths[0]).reshape((10000,28,28,1))
train_imgs = readimgs(filepaths[2]).reshape((60000,28,28,1))



import mytorch as nn

class myMNIST(nn.Graph):
    def __init__(self):
        super().__init__()
        self.input = nn.Variable(self)
        self.conv1 = nn.conv2d(self,(4,4,1), 3, stride=2)  # (N,28,28,1) -> (N,13,13,3)
        self.batch_norm1 = nn.barch_norm(self,3)
        self.relu1 = nn.leaky_relu(self,0.1)
        self.conv2 = nn.conv2d(self,(3,3,3), 5, stride=2)  # (N,13,13,3) -> (N,6,6,5)
        self.batch_norm2 = nn.barch_norm(self,5)
        self.relu2 = nn.leaky_relu(self,0.1)
        self.flatten = nn.view(self,[6*6*5])
        self.linear1 = nn.linear(self,6*6*5,30)
        self.relu3 = nn.leaky_relu(self,0.1)
        self.linear2 = nn.linear(self,30,10)
        
        self.loss = nn.loss_softmax_cross_entropy(self)
        self.optimizer = nn.optim_simple(0.01)
        
    def forward(self,x,y):
        self.input.connect(x)
        self.conv1.connect(self.input)
        self.batch_norm1.connect(self.conv1, self.infer)
        self.relu1.connect(self.batch_norm1)
        self.conv2.connect(self.relu1)
        self.batch_norm2.connect(self.conv2, self.infer)
        self.relu2.connect(self.batch_norm2)
        self.flatten.connect(self.relu2)
        self.linear1.connect(self.flatten)
        self.relu3.connect(self.linear1)
        self.linear2.connect(self.relu3)
        self.loss.connect(self.linear2, y)
        
        self.outputs = np.argmax(self.loss.outputs,axis=-1)  # classification results from softmax


model = myMNIST()
batch_size = 20
epoch = 10000

t_start = time.time()

for i in range(epoch):
    idxes = np.random.randint(60000, size=batch_size)
    x = train_imgs[idxes]
    y = train_labels[idxes]
    
    model.forward(x,y)
    model.backward()
    model.step()
    if np.isnan(model.loss.loss):
        break
    if i % 100 == 99:
        print("iteration =",i,"\t\tloss =",model.loss.loss, "\t\tbatch_accuracy =",np.sum(model.outputs==y)/batch_size)

print("training time:",time.time()-t_start)

print(model.batch_norm1.parameters, model.batch_norm1.global_mean, model.batch_norm1.global_var)
print(model.batch_norm2.parameters, model.batch_norm2.global_mean, model.batch_norm2.global_var)

test_size = 9000
idxes = np.random.randint(10000, size=test_size)
x = test_imgs[idxes]
y = test_labels[idxes]
model.infer = True
model.forward(x,y)

accuracy = np.sum(model.outputs == y) / test_size
print("test accuracy:",accuracy)