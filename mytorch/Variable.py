"""
Pytorch-like autograd computation graph
"""

import numpy as np 

class Variable:
    def __init__(self):
        self.prevs = []          # prev nodes
        self.prevs_grads = []    # gradients for backpropogation, dim[0]=batch_size
        self.parameters = []     # parameters of this node
        self.grads = []          # gradients of parameters, same shape of parameters
        self.n_next = 0          # fan_out
        self.outputs = None      # curr outputs, a tensor whose dim[0]=batch_size
        self.need_grad = False   # if no parameters, does not need grad
        self.print_info =  False # print something
    def connect(self, x):        # construct the graph and calculate the outputs 
        self.outputs = x
    def autograd(self, grads_in):# calculate grads of parameters and inputs, and do backpropogation
        return
    def step(self, length):      # update parameters
        if(self.need_grad):
            for i in range(len(self.parameters)):
                self.parameters[i] -= length*(self.grads[i])
                # self.parameters[i][self.parameters[i]>10] = 10     # set the threshold
                # self.parameters[i][self.parameters[i]<-10] = -10
        for i in range(len(self.prevs)): 
            self.prevs[i].step(length)


 
class Graph:
    def __init__(self):
        self.outputs = None      # forward output
        self.loss = None        # loss function node
    def backward(self):
        self.loss.autograd(1)
    def step(self,length):
        self.loss.step(length)