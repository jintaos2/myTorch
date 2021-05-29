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
def get_onehot(input, n_class):
    ret = np.zeros((len(input),n_class))
    for i in range(len(input)):
        ret[i][input[i]] = 1 
    return ret
test_labels = get_onehot(readlabels(filepaths[1]),10)
train_labels = get_onehot(readlabels(filepaths[3]),10)
def readimgs(path)->np.ndarray:
    with gzip.open(path, 'rb') as f1:
        data = f1.read()
        n_imgs = (len(data)-16)//(28*28)  # 舍弃前16个字符
        fmt = '>{}B'.format(28*28)        # > 表示大端模式， 读取28*28个8位像素
        images = np.empty((n_imgs,28,28))
        for i in range(n_imgs):
            offset = 16+28*28*i
            images[i] = np.array(struct.unpack_from(fmt, data, offset)).reshape((28,28))
    return images
# 10000, 28*28
test_imgs = readimgs(filepaths[0]).reshape((10000,-1))
train_imgs = readimgs(filepaths[2]).reshape((60000,-1))



import mytorch as nn
nn.setLogger('./')

batch_size = 15
epoch = 20000

class myMNIST(nn.Graph):
    def __init__(self):
        super().__init__()
        # feed
        self.x: nn.Port = nn.Port(np.zeros((1,28*28)))   # fake batch size
        self.y: nn.Port = nn.Port(np.zeros((1,10)))
        # layers
        self.linear1 = nn.linear(self, 40, self.x)
        self.linear1_out: nn.Port = self.linear1.outputs[0]
        
        self.relu1 = nn.leaky_relu(self,0.1, self.linear1_out)
        self.relu1_out: nn.Port = self.relu1.outputs[0]
        
        self.linear2 = nn.linear(self,30, self.relu1_out)
        self.linear2_out: nn.Port =  self.linear2.outputs[0]
        
        self.relu2 = nn.leaky_relu(self,0.1,self.linear2_out)
        self.relu2_out: nn.Port = self.relu2.outputs[0]
        
        self.linear3 = nn.linear(self,10, self.relu2_out)
        self.linear3_out = self.linear3.outputs[0]
        # loss
        self.loss = nn.loss_softmax_cross_entropy(self, self.linear3_out, self.y)
        self.loss_out = self.loss.outputs[1]
        self.outputs = self.loss.outputs[0]
        
        self.optimizer = nn.optim_simple(0.01)
    """
    x: (batch size, 28*28)
    y: (batch size, 10)
    """
    def feed(self, x:np.ndarray, y:np.ndarray):
        self.x.value = x 
        self.y.value = y
        
model = myMNIST()

t_start = time.time()
for i in range(epoch):
    
    idxes = np.random.randint(60000, size=batch_size)
    x = train_imgs[idxes]
    y = train_labels[idxes]
    
    model.feed(x,y)
    model.forward()
    model.backward()
    model.step()
    loss = model.loss_out.extra
    if np.isnan(loss):
        break
    if i % 100 == 99:
        nn.Logger.info(f"iteration = {i}\t\tloss={loss}" )

nn.Logger.info("training time: {}".format(time.time()-t_start))

x = test_imgs
y = test_labels
model.feed(x,y)
model.forward()

accuracy = np.sum(model.outputs.value.argmax(axis=-1) == y.argmax(axis=-1)) / 10000
nn.Logger.info(f"test accuracy: {accuracy}")
