"""
Pytorch-like autograd computation graph

1 to n node: gradients are accumulated n times, then backward 
n to 1 node: back-propagation is called n times
n to m node: gradients are accumulated m times, then call back-propagation n times

定义网络时需要满足有向无环图的拓扑排序
"""

import numpy as np
import logging
import os
from typing import List, Set, Dict, Tuple, Optional

Logger = logging.getLogger()
def setLogger(path):
    global Logger
    Logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s (%(filename)s)[line:%(lineno)d] %(message)s",'%m/%d %I:%M:%S')
    # file
    fh = logging.FileHandler(os.path.join(path, "log"), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    Logger.addHandler(fh)
    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    Logger.addHandler(ch)
    
    

class Port:
    """
    input and output port, part of Node, 
    contains one tensor, and its gradient
    """
    def __init__(self, input: np.ndarray, extra = None):
        self.value: np.ndarray = input         
        # backward gradient updated by next node in back-propagation
        self.grad: np.ndarray = np.zeros(input.shape)   
        # extra parameters
        self.extra = extra   
        
        
class Node:
    """
    graph: father graph 
    ports: reference of ports of prev nodes
    """
    def __init__(self, graph, *ports:Port):  
        
        graph.nodes.append(self)             # add node to graph, in order

        self.inputs: List[Port] = [*ports]   # input ports, can be empty
        self.outputs: List[Port] = []        # output ports

        # optimizer will use
        self.parameters: List[np.ndarray] = []     # parameters of this node, can be empty
        self.gradients: List[np.ndarray] = []      # gradients of parameters of this node
    """    
    forward: update output
    backward: calculate grads of parameters and inputs
    ! batch_size is variable !
    """
    def forward(self):      
        return
    def backward(self):     
        return


class Graph:
    def __init__(self):
        
        self.nodes: List[Node] = []     # all graph nodes, should in proper order for forwarding
        self.optimizer = None           # optimizer
         
        self.outputs:Port = None        # not necessary      

    def feed(self):                     # feed data, set some parameters manually
        return 
    def forward(self):
        for i in self.nodes:
            i.forward()
    def backward(self):
        for i in reversed(self.nodes):
            i.backward()
    def step(self):
        for i in self.nodes:
            self.optimizer.step(i)


class optim_simple:
    """
    a simple optimizer
    """
    def __init__(self, length:float, threshold=1e20):
        self.length = length
        self.threshold = threshold

    def step(self, node:Node):
        for i in range(len(node.parameters)):  
            node.parameters[i] -= self.length*(node.gradients[i])
            node.parameters[i] = np.clip( node.parameters[i], -self.threshold, self.threshold)
