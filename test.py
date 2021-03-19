
import numpy as np


class Variable:
    def __init__(self,graph): 
        graph._nodes.append(self)
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