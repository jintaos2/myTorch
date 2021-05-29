
from .Graph import *
from typing import List, Set, Dict, Tuple, Optional

"""
this kind of node will add up grads until it is called by all reference nodes
"""
class branch(Node):
    def __init__(self, graph):                               
        super().__init__(graph)
    def connect(self, x):
        self.prevs = [x]          
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        self.n_called = 0
        x.n_next += 1                             
        self.outputs = x.outputs              
    def autograd(self,grads_in): 
        self.prevs_grads[0] += grads_in
        self.n_called += 1
        if self.n_called == self.n_next:
            self.n_called = 0
            self.prevs[0].autograd(self.prevs_grads[0])    
"""            
reshape, ignore the batch_size
input: list [h,w,...]
"""
class view(Node):
    def __init__(self, graph:Graph, output_shape, *ports:Port):                               
        graph.nodes.append(self)
        self.inputs: List[Port] = [*ports]    
        self.input_shape = self.inputs[0].value.shape
        self.output_shape = [self.input_shape[0], *output_shape]   
        self.outputs: List[Port] = [ Port(np.zeros(self.output_shape)) ]
        self.parameters: List[np.ndarray] = []     
        self.gradients: List[np.ndarray] = []  
    def forward(self):
        self.outputs[0].value = self.inputs[0].value.reshape(self.output_shape) 
    def backward(self):
        self.inputs[0].grad = self.outputs[0].grad.reshape(self.input_shape)


"""
input:  graph, and a port(a tensor) 
output: exp(input tensor)
input shape: any
"""
class exp(Node):
    def __init__(self, graph:Graph, *ports:Port):                               
        graph.nodes.append(self)
        self.inputs: List[Port] = [*ports]            # reference of ports of prev nodes
        self.outputs: List[Port] = [Port(np.zeros(self.inputs[0].value.shape)) ] # output shape = input shape
        self.parameters: List[np.ndarray] = []     # no parameters
        self.gradients: List[np.ndarray] = []      # no gradients 
    def forward(self):
        self.outputs[0].value = np.exp(self.inputs[0].value)  # update output value
    def backward(self):
        self.inputs[0].grad = self.outputs[0].grad * self.outputs[0].value  # update grads
        


class sigmoid(Node):
    def __init__(self, graph:Graph, *ports:Port):                               
        graph.nodes.append(self)
        self.inputs: List[Port] = [*ports]             
        self.outputs: List[Port] = [ Port(np.zeros(self.inputs[0].value.shape)) ]
        self.parameters: List[np.ndarray] = []     
        self.gradients: List[np.ndarray] = []    
    def forward(self):
        self.outputs[0].value = 1.0/(1+np.exp(self.inputs[0].value))  
    def backward(self):
        y = self.outputs[0]
        self.inputs[0].grad = y.grad * y.value * (1-y.value)
        

class relu(Node):
    def __init__(self, graph:Graph, *ports:Port):                               
        graph.nodes.append(self)
        self.inputs: List[Port] = [*ports]             
        self.outputs: List[Port] = [ Port(np.zeros(self.inputs[0].value.shape)) ]  
        self.parameters: List[np.ndarray] = []     
        self.gradients: List[np.ndarray] = []    
    def forward(self):
        x = self.inputs[0]
        self.outputs[0].value = x.value * (x.value > 0)  
    def backward(self):
        y = self.outputs[0]
        self.inputs[0].grad = y.grad * (y.value > 0)
        

        
class leaky_relu(Node):
    """
    input: (batch size, ...)
    output: (batch size, ...)
    """
    def __init__(self, graph:Graph, coeff:float, inputs:Port):                               
        graph.nodes.append(self)
        self.inputs:  List[Port] = [inputs]             
        self.outputs: List[Port] = [ Port(np.zeros(inputs.value.shape))]  
        self.parameters: List[np.ndarray] = []     
        self.gradients:  List[np.ndarray] = []    
        self.coeff = coeff   # 0.1  
    def forward(self):
        x = self.inputs[0]
        y = self.outputs[0]
        self.local_grads = (x.value < 0) * self.coeff + (x.value >= 0) * 1.0
        y.value = x.value * self.local_grads  
    def backward(self):
        x = self.inputs[0]
        y = self.outputs[0]
        x.grad = self.local_grads * y.grad   # y > 0 ? 1 : 0.1
        

class linear(Node):
    """
    last dim -> new dim: tensordot([(-1),(-1)])
    (batch_size, input_size)*(output_size,input_size)+(1,output_size) = (batch_size, output_size)
    """
    def __init__(self, graph:Graph, output_size:int, inputs:Port):                               
        graph.nodes.append(self)
        self.inputs:  List[Port] = [inputs]             
        batch_size, input_size = inputs.value.shape    # fake batch_size 
        self.outputs: List[Port] = [ Port(np.zeros((batch_size, output_size))) ]  
        self.parameters: List[np.ndarray] = [(np.random.random((output_size, input_size))-0.5)/1000.0, 
                                             (np.random.random((output_size))-0.5)/1000.0]    
        self.gradients:  List[np.ndarray] = [np.zeros((self.parameters[0].shape)), np.zeros((self.parameters[1].shape))]
    def forward(self):
        x = self.inputs[0]
        y = self.outputs[0]
        y.value = np.tensordot(x.value, self.parameters[0], axes=[(-1),(-1)]) + self.parameters[1]
    def backward(self):
        y = self.outputs[0] 
        x = self.inputs[0]
        self.gradients[0] = np.tensordot( y.grad, x.value, axes=[(0), (0)])   # (output_size,input_size)
        self.gradients[1] = np.sum(y.grad,axis=0)
        x.grad = np.tensordot(y.grad, self.parameters[0], axes=[(-1),(0)])
        

class embedding(Node):
    """
    行向量，axes 从左往右递增
    (batch_size,seq_size,dict_size)*(dict_size,embedding_size) = (batch_size, seq_size, embedding_size)
    """
    def __init__(self, graph, dict_size, embedding_size, position_encoding_mat:np.ndarray):    
        super().__init__(graph)
        # initialize the weight and bias
        self.parameters=[np.random.random((dict_size,embedding_size))/1000.0] 
        self.need_grad = True
        self.position_encoding_mat = position_encoding_mat
    def connect(self, x):
        self.prevs = [x]   
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        self.grads = [np.zeros((self.parameters[0].shape))]
        x.n_next += 1
        self.outputs = np.tensordot(x.outputs,self.parameters[0], axes=[(2),(0)]) + self.position_encoding_mat 
    def autograd(self,grads_in):  
        """
        grads_in: (batch_size, seq_size, embedding_size)
        prev_output: (batch_size,seq_size,dict_size)
        curr_parameter: (dict_size,embedding_size)
        """
        self.grads[0] += np.tensordot(self.prevs[0].outputs,grads_in, axes=[(0,1),(0,1)])       
        self.prevs_grads[0] += np.tensordot(grads_in, self.parameters[0], axes=[(2),(1)])   
        self.prevs[0].autograd(self.prevs_grads[0])

"""
CONV
"""

# reshape, padding, input_size = (N, H, W, C)   
class padding(Node):
    def __init__(self, graph, h = (0,0), w=(0,0), value = 0):                               
        super().__init__(graph)
        self.h = h
        self.w = w
    def connect(self, x):
        self.prevs = [x]          
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        x.n_next += 1                       
        self.outputs = np.pad(x.outpust,((0,0),self.h,self.w,(0,0)))            
    def autograd(self,grads_in): 
        self.prevs_grads[0] += np.pad(grads_in,((0,0),self.h,self.w,(0,0)))   
        self.prevs[0].autograd(self.prevs_grads[0]) 



def im2col_strides(x, Hk, Hw, s):
    N, H, W, C = x.shape
    H_out = (H - Hk) // s + 1                      # output width 
    W_out = (W - Hw) // s + 1                      # output height
    shape = (N, H_out, W_out, Hk, Hw, C)
    # start: X.strides[0], X.strides[1]*s, X.strides[2]*s
    # block: shape = (Hk,Hw,C) , stride = X.strides[H,W,C]
    strides = (x.strides[0], x.strides[1]*s, x.strides[2]*s, *x.strides[1:])
    ret = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return ret
"""
used in back-propagation of conv2d 
"""
def pad_strides(x, padding = (0,0), stride = 1):
    if stride == 1:
        return np.pad(x,((0,0),(padding[0],padding[0]),(padding[1],padding[1]),(0,0)))
    N,H,W,C = x.shape 
    H_out = (H-1)*stride+1 + 2*padding[0]
    W_out = (W-1)*stride+1 + 2*padding[1]
    ret = np.zeros((N,H_out,W_out,C))
    ret[:,padding[0]:H_out-padding[0]:stride,padding[1]:W_out-padding[1]:stride,:] = x
    return ret


class conv2d(Node):
    """
    kernel_shape: Hk, Wk, C_in 

    out_channels: C_out

    input_size: N, H, W, C

    notice: no padding, size should fit! H_out = (H_in - Hk)/stride + 1
    """
    def __init__(self, graph, kernel_shape, out_channels, stride=1):
        super().__init__(graph)
        self.parameters=[np.random.random((out_channels,*kernel_shape))/1000.0, np.random.random((out_channels))/1000.0] 
        self.need_grad = True   
        self.stride = stride   
        self.Hk,self.Wk,_ = kernel_shape

    def connect(self, x):
        self.prevs = [x]   
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        self.grads = [np.zeros((self.parameters[0].shape)), np.zeros((self.parameters[1].shape))]
        x.n_next += 1
        self.inputs = x.outputs
        self.x_im2col = im2col_strides(self.inputs,self.Hk,self.Wk,self.stride)  # (N, H_out, W_out, Hk, Wk, C_in)
        # (N, H_out, W_out, Hk, Wk, C_in) * (C_out, Hk, Wk, C_in) -> (N, H_out, W_out, C_out)
        self.outputs = np.tensordot(self.x_im2col, self.parameters[0], axes=[(3,4,5), (1,2,3)]) + self.parameters[1]    

    def autograd(self,grads_in):  # grads_in: (N, H_out, W_out, C_out)
        self.grads[1] += np.sum(grads_in, axis=(0,1,2))    # (N, H_out, W_out, C_out) -> (C_out)
        # (N, H_out, W_out, C_out) * (N, H_out, W_out, Hk, Wk, C_in) -> (C_out, Hk, Wk, C_in)
        self.grads[0] += np.tensordot(grads_in, self.x_im2col, axes=[(0,1,2), (0,1,2)]) 
        # new kernel reshape: (C_in, Hk, Wk, C_out) and rotate 180 deg
        kernel_ = np.flip(self.parameters[0],(1,2)).swapaxes(0,3)
        grad_pad = pad_strides(grads_in, padding=(self.Hk-1,self.Wk-1), stride=self.stride)
        # print(grad_pad.shape)
        grad_im2col = im2col_strides(grad_pad,self.Hk,self.Wk,1)     # notice that the stride = 1
        # (N, H_new, W_new, Hk, Wk, C_out) * (C_in, Hk, Wk, C_out) -> (N, H, W, C_in)
        # H_new == H, W_new == W ?
        # print(grad_im2col.shape, kernel_.shape, self.prevs_grads[0].shape)
        self.prevs_grads[0] += np.tensordot(grad_im2col, kernel_, axes=[(3,4,5), (1,2,3)]) 
        self.prevs[0].autograd(self.prevs_grads[0])

class barch_norm(Node):
    def __init__(self, graph, C, decay=0.95, eps = 1e-5):                               
        super().__init__(graph)
        self.parameters=[np.ones(C), np.zeros(C)]  # gamma, beta
        self.need_grad = True
        self.decay = decay 
        self.eps = eps 
        self.global_mean = 0
        self.global_var = 1
    # using global mean and variance while inference
    def connect(self, x, infer = False):
        self.prevs = [x]          
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        self.grads = [np.zeros((self.parameters[0].shape)), np.zeros((self.parameters[1].shape))]
        x.n_next += 1   
        mean = np.mean(x.outputs, axis=(0,1,2))   # (N,H,W,C)
        var =  np.var(x.outputs, axis=(0,1,2)) 
        self.global_mean = self.global_mean * self.decay + mean*(1-self.decay)
        self.global_var = self.global_var * self.decay + var*(1-self.decay)
        if infer:
            self.yy = (x.outputs - self.global_mean) / np.sqrt(self.global_var + self.eps)
        else:
            self.yy = (x.outputs - mean) / np.sqrt(var + self.eps)
        self.N = x.outputs.shape[0] * x.outputs.shape[1] * x.outputs.shape[2]
        self.coef = self.parameters[0] / self.N / np.sqrt(var + self.eps)
        self.outputs = self.yy*self.parameters[0] + self.parameters[1]           
    def autograd(self,grads_in): 
        self.grads[0] += np.sum(grads_in * self.yy, axis=(0,1,2))  # gamma
        self.grads[1] += np.sum(grads_in, axis=(0,1,2))   # beta
        self.prevs_grads[0] +=  self.coef * (self.N*grads_in - self.grads[1]-self.yy*self.grads[0])
        self.prevs[0].autograd(self.prevs_grads[0]) 
