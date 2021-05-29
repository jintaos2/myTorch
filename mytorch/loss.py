from .Graph import *

class loss_MSE(Node):
    def __init__(self, graph):      
        super().__init__(graph)
    def connect(self, x, y):   # 2 inputs, 1D array
        self.prevs = [x]    
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        x.n_next += 1
        self.x_ = x.outputs 
        self.y_ = np.array(y.outputs).reshape(x.outputs.shape)
        self.outputs =  np.mean((self.x_-self.y_)**2)
    def autograd(self,grads_in): 
        self.prevs_grads[0] += grads_in*2.0/self.x_.shape[0]*(self.x_ - self.y_)
        self.prevs[0].autograd(self.prevs_grads[0])
        
        
        
    # def __init__(self, graph:Graph, *ports:Port):                               
    #     graph.nodes.append(self)
    #     self.inputs: tuple[Port] = ports            
    #     self.outputs: tuple[Port] = ( Port(np.zeros(self.inputs[0].value.shape)) )  
    #     self.parameters: list[np.ndarray] = []     
    #     self.gradients: list[np.ndarray] = []    
    # def forward(self):
    #     self.outputs[0].value = 1.0/(1+np.exp(self.inputs[0].value))  
    # def backward(self):
    #     y = self.outputs[0]
    #     self.inputs[0].grad = y.grad * y.value * (1-y.value)
class loss_softmax_cross_entropy(Node):
    """
    input:   (batch_size, ...,  n_classes) x 2   
    output:  (batch_size, ..., n_classes), and a loss value
    
    input[0] is prev output 
    input[1] is ground truth (onehot encoding)
    output[0] is class probability
    output[1] is loss
    """
    def __init__(self, graph:Graph, y_out:Port, y_true:Port):                               
        graph.nodes.append(self)
        self.inputs: List[Port] = [y_out, y_true]      
        self.outputs: List[Port] = [Port(np.zeros(y_out.value.shape)), Port(np.zeros(1), 0.0)]
        self.parameters: List[np.ndarray] = []     
        self.gradients: List[np.ndarray] = []  
    def forward(self):
        x = self.inputs[0]
        shape1 = [*x.value.shape]
        self.batch_size = shape1[0]
        shape1[-1] = 1                     #(batch_size, ..., 1)
        shape1 = tuple(shape1)
        shifted = np.exp(x.value - np.max(x.value,axis=-1).reshape(shape1))       # avoid overflow
        self.outputs[0].value = shifted / np.sum(shifted,axis=-1).reshape(shape1) # softmax output
        loss1 = self.outputs[0].value * self.inputs[1].value
        loss1 = np.sum(loss1,axis=-1) + 1e-30
        loss = -np.sum(np.log(loss1))           # loss output
        self.outputs[1].extra = loss/self.batch_size
        
    def backward(self):
        self.inputs[0].grad = (self.outputs[0].value - self.inputs[1].value)/self.batch_size # backward grad
        
    # def connect(self, x, y):   
    #     self.prevs = [x]    
    #     self.prevs_grads = [np.zeros(x.outputs.shape)]
    #     x.n_next += 1
    #     self.batch_size = x.outputs.shape[0]
    #     temp = np.exp(x.outputs-np.max(x.outputs,axis=-1).reshape((self.batch_size,1)))
    #     self.outputs = temp / np.sum(temp,axis=-1).reshape((self.batch_size,1))
    #     self.y_ = np.array(y).reshape((-1))
    #     self.loss = 0
    #     for i in range(self.batch_size):
    #         self.loss -= np.log(self.outputs[i][self.y_[i]]+1e-30)
    #     self.loss /= self.batch_size
    # def autograd(self,grads_in): 
    #     grad_ = self.outputs.copy()
    #     for i in range(self.batch_size):
    #         grad_[i][self.y_[i]] -= 1.0
    #     self.prevs_grads[0] += grad_ / self.batch_size
    #     self.prevs[0].autograd(self.prevs_grads[0])