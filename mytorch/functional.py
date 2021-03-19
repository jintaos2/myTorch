
from .Variable import *
# this kind of node will add up grads until it is called by all reference nodes
class branch(Variable):
    def __init__(self):                               
        super().__init__()
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
# reshape, ignore the batch_size
# input: list [h,w,...]
class view(Variable):
    def __init__(self, output_shape):                               
        super().__init__()
        self.output_shape = output_shape
    def connect(self, x):
        self.prevs = [x]          
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        x.n_next += 1  
        self.batch_size = x.outputs.shape[0]                         
        self.outputs = x.outputs.reshape((self.batch_size, *self.output_shape))              
    def autograd(self,grads_in): 
        self.prevs_grads[0] += grads_in.reshape(self.prevs_grads[0].shape)
        self.prevs[0].autograd(self.prevs_grads[0])  

class exp(Variable):
    def __init__(self):                               # no parameters needed
        super().__init__()
    def connect(self, x):
        self.prevs = [x]                              # store the prev nodes
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        x.n_next += 1                                 # ref count
        self.outputs = np.exp(x.outputs)              # calculate output
    def autograd(self,grads_in): 
        self.prevs_grads[0] += grads_in * self.outputs
        self.prevs[0].autograd(self.prevs_grads[0])


class sigmoid(Variable):
    def __init__(self):      
        super().__init__()
    def connect(self, x):
        self.prevs = [x]     
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        x.n_next += 1
        self.outputs = 1.0/(1+np.exp(x.outputs)) 
    def autograd(self,grads_in): 
        self.prevs_grads[0] += grads_in * self.outputs * (1-self.outputs)
        self.prevs[0].autograd(self.prevs_grads[0])
        
   
class relu(Variable):
    def __init__(self):     # no parameters needed
        super().__init__()
    def connect(self, x):
        self.prevs = [x]    
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        x.n_next += 1
        self.outputs = x.outputs * (x.outputs > 0) 
    def autograd(self,grads_in): 
        self.prevs_grads[0] += grads_in * (self.outputs>0) * 1.0
        self.prevs[0].autograd(self.prevs_grads[0])
class leaky_relu(Variable):
    def __init__(self,a):    
        super().__init__()
        self.a = -a
    def connect(self, x):
        self.prevs = [x]  
        x.n_next += 1
        self.local_grads = (x.outputs < 0)*self.a + (x.outputs >= 0) * 1.0
        self.outputs = x.outputs*self.local_grads
        self.prevs_grads = [np.zeros(x.outputs.shape)]
#         if self.print_info:
#             print("leaky_relu\t - in:{} out:{}".format(x.outputs.shape,x.outputs.shape)) 
    def autograd(self,grads_in): 
        self.prevs_grads[0] += self.local_grads * grads_in
        self.prevs[0].autograd(self.prevs_grads[0])

class linear(Variable):
    """
    (batch_size,input_size)*(input_size,output_size) + (1,output_size) = (batch_size, output_size)
    """
    def __init__(self, input_size, output_size):    
        super().__init__()
        # initialize the weight and bias
        self.parameters=[np.random.random((input_size,output_size))/1000.0, np.random.random((output_size))/1000.0] 
        self.need_grad = True
    def connect(self, x):
        self.prevs = [x]   
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        self.grads = [np.zeros((self.parameters[0].shape)), np.zeros((self.parameters[1].shape))]
        x.n_next += 1
        # print(  x.outputs.shape, self.parameters[0].shape, self.parameters[1].shape)
        self.outputs = np.tensordot( x.outputs,self.parameters[0], axes=[(1),(0)]) + self.parameters[1]        # (N, W_in) * (W_in,W_out) -> (N,W_out)
    def autograd(self,grads_in):  
        self.grads[0] += np.tensordot(self.prevs[0].outputs,grads_in, axes=[(0),(0)])       # (N, W_in) * (N,W_out) -> (W_in, W_out)
        self.grads[1] += np.sum(grads_in,axis=0)
        self.prevs_grads[0] += np.tensordot(grads_in, self.parameters[0], axes=[(1),(1)])   # (N,W_out) * (W_in,W_out) ->  (N,W_in)
        if self.print_info:
            print("Grad_linear\t - self:{} back:{}".format(self.grads[0].shape,self.prevs_grads[0].shape))
            print("linear:\t",np.max(self.prevs_grads[0]))
        self.prevs[0].autograd(self.prevs_grads[0])

"""
CONV
"""


# reshape, padding, input_size = (N, H, W, C)   
class padding(Variable):
    def __init__(self, h = (0,0), w=(0,0), value = 0):                               
        super().__init__()
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


class conv2d(Variable):
    """
    kernel_shape: Hk, Wk, C_in 

    out_channels: C_out

    input_size: N, H, W, C

    notice: no padding, size should fit! H_out = (H_in - Hk)/stride + 1
    """
    def __init__(self, kernel_shape, out_channels, stride=1):
        super().__init__()
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
        self.grads[1] += np.sum(grads_in, axis=(0,1,2))                                 # (N, H_out, W_out, C_out) -> (C_out)
        # (N, H_out, W_out, C_out) * (N, H_out, W_out, Hk, Wk, C_in) -> (C_out, Hk, Wk, C_in)
        self.grads[0] += np.tensordot(grads_in, self.x_im2col, axes=[(0,1,2), (0,1,2)]) 
        # new kernel reshape: (C_in, Hk, Wk, C_out) and rotate 180 deg
        kernel_ = np.flip(self.parameters[0],(1,2)).swapaxes(0,3)
        grad_pad = pad_strides(grads_in, padding=(self.Hk-1,self.Wk-1), stride=self.stride)
        #print(grad_pad.shape)
        grad_im2col = im2col_strides(grad_pad,self.Hk,self.Wk,1)     # notice that the stride = 1
        # (N, H_new, W_new, Hk, Wk, C_out) * (C_in, Hk, Wk, C_out) -> (N, H, W, C_in)
        # H_new == H, W_new == W ?
        #rint(grad_im2col.shape, kernel_.shape, self.prevs_grads[0].shape)
        self.prevs_grads[0] += np.tensordot(grad_im2col, kernel_, axes=[(3,4,5), (1,2,3)]) 
        self.prevs[0].autograd(self.prevs_grads[0])

class barch_norm(Variable):
    def __init__(self, C, decay=0.95, eps = 1e-5):                               
        super().__init__()
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
