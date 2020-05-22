# Imports, set pytorch grad enabled to false according to instructions

import torch
import dlc_practical_prologue as prologue
import math
torch.set_grad_enabled(False)


class Module ( object ) :
    def __call__(self, *args, **kwargs):
        return self.forward(*args)
    def forward ( self , * input ) :
        raise NotImplementedError
    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError
    def param ( self ) :
        return []
    
    

'''
Applies a linear transformation to the incoming data: y = xA^T + b

Args:
    in_features: size of each input sample
    out_features: size of each output sample
    bias: If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``

Attributes:
    weight: the learnable weights of the module 
    bias:   the learnable bias of the module of shape 


'''
class Linear(Module):

    def __init__(self, in_features, out_features, bias = True):
        self.in_features = in_features
        self.out_features = out_features

        
        # Initialize weight and bias
        # according to https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
        
        if bias:
            self.bias = torch.empty(1,out_features)
        self.weight = torch.empty(in_features,out_features)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
        # cache the input whenever we go forward
        self.x = 0
        self.grad_weight = torch.empty(in_features,out_features)
        self.grad_bias = torch.empty(1,out_features)
            
        
    #Applies a linear transformation to the incoming data: y = xA^T + b   
    def forward(self,input):
        self.x = input
        output = input.mm(self.weight)
        if self.bias is not None:
            output += self.bias

        return output


    def backward(self,input):
        x = self.x
        w = self.weight
        b = self.bias
        
        # divide weigthts by batch size 
        # inspiration from https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
        db = input.sum(0).div(input.size()[0])
        dx = input.mm(w.t())
        dw = x.t().mm(input).div(input.size()[0])
        
        

        self.grad_bias += db
        self.grad_weight += dw


        return dx
            

    def param(self):
        return[(self.weight,self.grad_weight),(self.bias,self.grad_bias)]


'''
Applies the element-wise function Rectified Linear Unit


'''
class ReLU(Module):
    def __init_(self):
        self.x = 0
    def forward( self,  input ):
        self.x = input
        s1 = input.clamp(min=0)
        return s1
    def backward(self, input):
        return (self.x>0).float()*input
    
        
        
    
'''
Applies the element-wise function Hyperbolic Tangent

'''

class Tanh(Module):
    def __init_(self):
        self.x = 0
        
    def forward(self,input):
        self.x = input
        s1 = input.tanh()
        return s1

    def backward(self, input):
            ds_dx = 1 - self.x.tanh().pow(2)
            dl_dx = ds_dx * input
            return dl_dx
        
'''
A sequential container.
Modules are added to it in the order they are passed in the constructor - in a list.

'''
class Sequential(Module):
    def __init__(self, param ):
        super().__init__()
        self.model = (param)
        
    def forward(self,x):
        for _ in self.model:
            x = _.forward(x)
        return x
    def backward(self,x):
        for _ in reversed(self.model):
            x = _.backward(x)
        return x
    
    def param(self):
        param_list = []
        for module in self.model:
            param_list.extend(module.param())

        return param_list

    
'''
Returns mean square error loss
Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input v and target t .
The reduction here is sum
'''

class MSELoss(Module):
    def forward(self,v,t):
        return (v-t).pow(2).sum(0).sum()
    def dloss(self,v,t):
        return (2*(v-t))
        
        
'''
To construct an Optimizer you have to give it an iterable containing the parameters (all should be Variable s) to optimize. Then, you can specify optimizer-specific options such as the learning rate.

'''
class SGD(object):

    def __init__(self, params, lr=0.01):
        self.lr = lr
        self.params = params


    def step(self):
        for _, (param, param_grad) in enumerate(self.params):
            # update parameter
            param.add_(-self.lr*param_grad)
    
    def zero_grad(self):
        # put all gradients to 0
        for param in self.params:
            param[1].zero_()