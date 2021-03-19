"""
Pytorch-like autograd computation graph
"""

import numpy as np


class Variable:
    print_info = False           # print something

    def __init__(self, graph):
        self.prevs = []          # prev nodes
        self.prevs_grads = []    # gradients for backpropogation, dim[0]=batch_size

        self.parameters = []     # parameters of this node
        self.grads = []          # gradients of parameters of this node
        self.outputs = None      # a tensor whose dim[0]=batch_size

        self.n_next = 0          # fan_out, used when this node is connected to many nodes
        self.need_grad = False   # if no parameters, does not need grad
        graph._nodes.append(self)

    def connect(self, x):        # construct the graph and calculate the outputs
        self.outputs = x         # default input is a tensor and output is input

    def autograd(self, grads_in):  # calculate grads of parameters and inputs, and do backpropogation
        return


class Graph:
    def __init__(self):
        self.outputs = None      # output tensor of this graph

        self.loss = None         # loss function, end of inference, start of backpropagation
        self.optimizer = None    # optimizer
        self._nodes = []         # list of all nodes
        
        self.infer = False       # used in batch_normalization

    def backward(self):
        self.loss.autograd(1)

    def step(self):
        for node in self._nodes:
            if node.need_grad:
                self.optimizer.step(node)


class optim_simple:
    def __init__(self, length, threshold=1e20):
        self.length = length
        self.threshold = threshold

    def step(self, node):
        for i in range(len(node.parameters)):  # numpy
            node.parameters[i] -= self.length*(node.grads[i])
            node.parameters[i] = np.clip( node.parameters[i], -self.threshold, self.threshold)
