
from .Graph import *
from typing import List, Set, Dict, Tuple, Optional

"""
1 port to n ports
"""
class branch(Node):
    def __init__(self, graph:Graph, N:int, inputs:Port):                               
        graph.nodes.append(self)
        self.inputs: List[Port] = [inputs]             
        self.forward()
        self.parameters: List[np.ndarray] = []     
        self.gradients: List[np.ndarray] = [] 
        self.N = N   
    def forward(self):
        self.outputs: List[Port] = []
        for i in range(self.N):
            self.outputs.append(copy.deepcopy(self.inputs[0]))
    def backward(self):
        x = self.inputs[0]
        x.grad = np.zeros((x.value.shape))
        for i in self.outputs:
            x.grad += i.grad
             
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
        
"""
transformer
"""
class embedding(Node):
    """
    (batch_size, seq_size, dict_size) * (embedding_size, dict_size) + position_mat = (batch_size, seq_size, embedding_size)
    position_mat: (seq_size, embedding_size), constant
    """
    def __init__(self, graph:Graph, embedding_size:int, position_mat:np.ndarray, inputs:Port):    
        graph.nodes.append(self)
        self.inputs:  List[Port] = [inputs]             
        batch_size, seq_size, dict_size = inputs.value.shape     
        self.outputs: List[Port] = [ Port(np.zeros((batch_size, seq_size, embedding_size))) ]  
        self.parameters: List[np.ndarray] = [(np.random.random((embedding_size, dict_size))-0.5)/1000.0]    
        self.gradients:  List[np.ndarray] = [np.zeros((self.parameters[0].shape)) ]
        self.position_mat = position_mat
    def forward(self):
        x = self.inputs[0]
        y = self.outputs[0]
        y.value = np.tensordot(x.value, self.parameters[0], axes=[(-1),(-1)]) + self.position_mat
    def backward(self):
        x = self.inputs[0]
        y = self.outputs[0]
        # Logger.info(f"x_value:{x.value.shape}  y_value:{y.value.shape}  y_grad:{y.grad.shape}")
        self.gradients[0] = np.tensordot( y.grad, x.value, axes=[(0,1), (0,1)])
        x.grad = np.tensordot(y.grad, self.parameters[0], axes=[(-1),(0)])


def softmax_grad(y:np.ndarray, grad_in:np.ndarray):
    shape1 = y.shape 
    length = shape1[-1]
    y_ = y.reshape((-1,length))
    grad_ = grad_in.reshape((-1,length))
    out = np.zeros(y_.shape)
    for i in range(y_.shape[0]):
        y_line = y_[i]
        grad_line = grad_[i]
        out[i] = np.dot(grad_line, np.diag(y_line) - np.outer(y_line, y_line))
    return out.reshape(shape1)
    
def matrix_grad(X, A, grad_in):
    """
    2d tensors: Y = np.dot(X,A)
    """
    grad_A = np.dot(X.T, grad_in)
    grad_X = np.dot(grad_in, A.T)
    return grad_X, grad_A

class attention(Node):
    """
    input: (batch_size, seq_size, embedding_size)
    W_Q: (embedding_size, width_qk) -> Q: (batch_size, seq_size, width_qk) 
    W_K: (embedding_size, width_qk) -> K: (batch_size, seq_size, width_qk)
    W_V: (embedding_size, embedding_size) -> V: (batch_size, seq_size, embedding_size)
    Q * K^T : (batch_size, seq_size, seq_size)
    softmax(Q * K^T + mask): (batch_size, seq_size, seq_size)
    Q * K^T * V: (batch_size, seq_size, embedding_size)
    """
    def __init__(self, graph:Graph, width_qk:int, mask:Port, inputs:Port):  
        graph.nodes.append(self)
        self.inputs:  List[Port] = [inputs, mask]
        self.outputs: List[Port] = [ Port(np.zeros(inputs.value.shape)) ]  
        _, self.seq_size, self.embedding_size = inputs.value.shape
        self.parameters: List[np.ndarray] = [(np.random.random((width_qk, self.embedding_size))-0.5)/1000.0,
                                             (np.random.random((width_qk, self.embedding_size))-0.5)/1000.0,
                                             (np.random.random((self.embedding_size, self.embedding_size))-0.5)/1000.0]    
        self.gradients:  List[np.ndarray] = [np.zeros((self.parameters[0].shape)), 
                                             np.zeros((self.parameters[1].shape)),
                                             np.zeros((self.parameters[2].shape))]
    def forward(self):
        x = self.inputs[0]
        y = self.outputs[0]
        self.batch_size = x.value.shape[0]
        self.Q = np.tensordot(x, self.parameters[0], axes=[(-1),(0)])
        self.K = np.tensordot(x, self.parameters[1], axes=[(-1),(0)])
        self.V = np.tensordot(x, self.parameters[2], axes=[(-1),(0)])
        self.QK = np.zeros((self.batch_size, self.seq_size, self.seq_size))
        for i in range(self.batch_size):
            self.QK[i] = np.dot(self.Q[i],self.K[i].T)/np.sqrt(self.embedding_size)
        self.QK -= np.max(self.QK, axis=-1).reshape((self.batch_size, self.seq_size, 1))  # avoid overflow
        self.QK_exp = np.exp(self.QK)
        self.QK_softmax = self.QK_exp/np.sum(self.QK_exp, axis=-1).reshape((self.batch_size, self.seq_size, 1))+self.inputs[1].value
        for i in range(self.batch_size):
            y.value[i] = np.dot(self.QK_softmax[i], self.V[i])
        
    def backward(self):
        x = self.inputs[0]
        y = self.outputs[0]
        x.grad = np.zeros(x.value.shape)
        self.gradients:  List[np.ndarray] = [np.zeros((self.parameters[0].shape)), 
                                             np.zeros((self.parameters[1].shape)),
                                             np.zeros((self.parameters[2].shape))]
        for i in range(self.batch_size):
            grad_softmax, grad_V = matrix_grad(self.QK_softmax[i], self.V[i], y.grad[i])
            grad_x1, grad_W_V = matrix_grad(x.value[i], self.parameters[2], grad_V)
            grad_QK = softmax_grad(self.QK_softmax[i], grad_softmax) / np.sqrt(self.embedding_size)
            grad_Q, grad_KT = matrix_grad(self.Q[i], self.K.T[i], grad_QK)
            grad_K = grad_KT.T 
            grad_x2, grad_W_K = matrix_grad(x.value[i], self.parameters[1], grad_K)
            grad_x3, grad_W_Q = matrix_grad(x.value[i], self.parameters[0], grad_Q)
            x.grad[i] = grad_x1 + grad_x2 + grad_x3 
            self.gradients[0] += grad_W_Q
            self.gradients[1] += grad_W_K
            self.gradients[2] += grad_W_V
            
class attention_encoder(Graph):
    def __init__(self,  graph:Graph,  width_qk:int, mask:np.ndarray, inputs:Port):
        graph.nodes.append(self)
        self.inputs:  List[Port] = [inputs]
        self.outputs: List[Port] = [ Port(np.zeros(inputs.value.shape)) ]
        
        self.atten1 = attention(self,width_qk, mask, inputs)
        
        
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
