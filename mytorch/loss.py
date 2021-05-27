from .Variable import *

class loss_MSE(Variable):
    def __init__(self, graph):      
        super().__init__(graph)
    def connect(self, x, y):   # 2 inputs, 1D array
        self.prevs = [x]    
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        x.n_next += 1
        self.x_ = x.outputs 
        self.y_ = np.array(y.outputs).reshape(x.outputs.shape)
        self.outputs =  np.mean((self.x_-self.y_)**2)
        if super().print_info:
            print("loss_MSE\t - in:{} out:{}".format(x.outputs.shape,self.outputs))
    def autograd(self,grads_in): 
        self.prevs_grads[0] += grads_in*2.0/self.x_.shape[0]*(self.x_ - self.y_)
        if super().print_info:
            print("Grad_loss_MSE\t - self:{} back:{}".format(None,self.prevs_grads[0].shape))
        self.prevs[0].autograd(self.prevs_grads[0])
        
class loss_softmax_cross_entropy(Variable):
    """
    input: (batch_size x n_classes)   output: (batch_size x n_classes)
    y is int labels
    """
    def __init__(self, graph):      
        super().__init__(graph)
    def connect(self, x, y):   
        self.prevs = [x]    
        self.prevs_grads = [np.zeros(x.outputs.shape)]
        x.n_next += 1
        self.batch_size = x.outputs.shape[0]
        temp = np.exp(x.outputs-np.max(x.outputs,axis=-1).reshape((self.batch_size,1)))
        self.outputs = temp / np.sum(temp,axis=-1).reshape((self.batch_size,1))
        self.y_ = np.array(y).reshape((-1))
        self.loss = 0
        for i in range(self.batch_size):
            self.loss -= np.log(self.outputs[i][self.y_[i]]+1e-30)
        self.loss /= self.batch_size
        if super().print_info:
            print("loss_softmax_cross_entropy\t - in:{} out:{}".format(x.outputs.shape,self.outputs.shape))
    def autograd(self,grads_in): 
        grad_ = self.outputs.copy()
        for i in range(self.batch_size):
            grad_[i][self.y_[i]] -= 1.0
        self.prevs_grads[0] += grad_ / self.batch_size
        if super().print_info:
            print("loss_softmax_cross_entropy\t - self:{} back:{}".format(None,self.prevs_grads[0].shape))
            print("loss_softmax_cross_entropy:\t",np.max(self.prevs_grads[0]))
        self.prevs[0].autograd(self.prevs_grads[0])