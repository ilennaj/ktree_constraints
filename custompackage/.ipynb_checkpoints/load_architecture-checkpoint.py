import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import custompackage.sl_custom as slc

from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch import functional as F

import numpy as np
import math



def kronecker(matrix1, matrix2):
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))

class NCK(nn.Module):

    def __init__(self, alpha=1, beta=0.6, gamma=1, learn=True, scale=1):
        super(NCK, self).__init__()
        if learn:
            self.alpha = Parameter(torch.tensor([float(alpha)]).requires_grad_()) # create a tensor out of alpha
            self.beta = Parameter(torch.tensor([float(beta)]).requires_grad_()) # create a tensor out of beta
            self.gamma = Parameter(torch.tensor([float(gamma)]).requires_grad_()) # create a tensor out of gamma
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
        self.scale = scale

    def forward(self, x):
        return(self.alpha*self.f_Na(x) + self.beta*self.f_Ca(x) + self.gamma*self.f_K(x))
    
    def f_Na(self, x, a=0.0878, b=113.68, c=6.39, d=8.98):
        x = x*self.scale
        return(a*(x-b)/(1+torch.exp((c - x/d))))

    def f_Ca(self, x, a=0.129, b=69.62, c=-4.40, d=4.25):
        x = x*self.scale
        return(a*(x-b)/(1+torch.exp((c - x/d))))

    def f_K (self, x, a=2.23, b=0.132, c=16.74, d=0.436):
        x = x*self.scale
        return(a/(d+torch.exp(-b*(x-c))))
    
class SQGL(nn.Module):

    def __init__(self, alpha=1, beta=0.6, gamma=1, learn=True, scale=1, atten=1, linscale=1):
        super(SQGL, self).__init__()
        self.learn = learn
        if learn:
            self.alpha = Parameter(torch.tensor([float(alpha)]).requires_grad_()) # create a tensor out of alpha
            self.beta = Parameter(torch.tensor([float(beta)]).requires_grad_()) # create a tensor out of beta
            self.gamma = Parameter(torch.tensor([float(gamma)]).requires_grad_()) # create a tensor out of gamma
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
        self.scale = scale
        self.atten = atten
        self.linscale = linscale

    def forward(self, x):
        if self.learn:
            self.alpha.data = torch.abs(self.alpha.data)
            self.beta.data = torch.abs(self.beta.data)
            self.gamma.data = torch.abs(self.gamma.data)
        I_ion = self.alpha*self.f_Na(x) + self.beta*self.f_Ca(x) + self.gamma*self.f_K(x)
        return((x*self.linscale + I_ion)*self.atten) #Attenuation 0.5

    
    def f_Na(self, x, a=0.0878, b=113.68, c=6.39, d=8.98):
        x = x*self.scale
        return(-a*(x-b)/(1+torch.exp((c - x/d).clamp(-60, 60))))

    def f_Ca(self, x, a=0.129, b=69.62, c=-4.40, d=4.25):
        x = x*self.scale
        return(-a*(x-b)/(1+torch.exp((c - x/d).clamp(-60, 60))))

    def f_K (self, x, a=2.23, b=0.132, c=16.74, d=0.436):
        x = x*self.scale
        return(-a/(d+torch.exp((-b*(x-c)).clamp(-60, 60))))
    

class Synapse(nn.Module):

    def __init__(self, in_features: int, Ep = 50, Em = -70, E0 = -65, learn: bool = True):
        super(Synapse, self).__init__()
        
        self.in_features = in_features
        # Note: Find reversal potential sources or choose new reversal potentials
        self.Ep = Ep 
        self.Em = Em
        self.E0 = E0 
        self.learn = learn
        
        # Initialize presynaptic activation dynamics(a1) and synapse size (a2, weights)
        self.register_parameter('ap1', Parameter(torch.Tensor(1, in_features)))
        self.register_parameter('ap2', Parameter(torch.Tensor(1, in_features)))
        self.register_parameter('am1', Parameter(torch.Tensor(1, in_features)))
        self.register_parameter('am2', Parameter(torch.Tensor(1, in_features)))
        self.register_parameter('g0', Parameter(torch.Tensor(1, in_features)))



        torch.nn.init.normal_(self.ap1, mean=0.0, std=math.sqrt(2/(self.ap1.shape[1])))
        torch.nn.init.normal_(self.ap2, mean=0.0, std=math.sqrt(2/(self.ap2.shape[1])))
        torch.nn.init.normal_(self.am1, mean=0.0, std=math.sqrt(2/(self.am1.shape[1])))
        torch.nn.init.normal_(self.am2, mean=0.0, std=math.sqrt(2/(self.am2.shape[1])))
        torch.nn.init.normal_(self.g0, mean=0.0, std=math.sqrt(2/(self.g0.shape[1])))
        

    def forward(self, x):
        top = self.g_n(x, self.ap1, self.ap2) * self.Ep + self.g_n(x, self.am1, self.am2) * self.Em + self.g0*self.E0
        bottom = self.g_n(x, self.ap1, self.ap2) + self.g_n(x, self.am1, self.am2) + self.g0 # Should I assume 1/R ~= 0?
        return(top / bottom)
    
    def g_n(self, x, a1, a2):
        return(torch.exp((a1 * x + a2).clamp(-60, 60)))

    
class F_Na(nn.Module):
    
    def __init__(self, scale=1):
        super(F_Na, self).__init__()
        self.scale = scale
        
    def forward(self, x, a=0.0878, b=113.68, c=6.39, d=8.98):
        x = x*self.scale
        return(-a*(x-b)/(1+torch.exp((c - x/d).clamp(-60, 60))))

class F_Ca(nn.Module):
    
    def __init__(self, scale=1):
        super(F_Ca, self).__init__()
        self.scale = scale
        
    def forward(self, x, a=0.129, b=69.62, c=-4.40, d=4.25):
        x = x*self.scale
        return(-a*(x-b)/(1+torch.exp((c - x/d).clamp(-60, 60))))
    
class F_K(nn.Module):
    
    def __init__(self, scale=1):
        super(F_K, self).__init__()
        self.scale = scale
        
    def forward(self, x, a=2.23, b=0.132, c=16.74, d=0.436):
        x = x*self.scale
        return(-a/(d+torch.exp((-b*(x-c)).clamp(-60, 60))))

class Hinge_loss(nn.Module):
    def __init__(self, margin = 1, reduction='mean'):
        super(Hinge_loss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, y, target):
        if self.reduction == 'sum':
            return(torch.sum(torch.max(torch.Tensor([0]).cuda(), self.margin - y*target)))
        elif self.reduction == 'mean':
            return(torch.mean(torch.max(torch.Tensor([0]).cuda(), self.margin - y*target)))
        else:
            return(torch.max(torch.Tensor([0]).cuda(), self.margin - y*target))
    
class simple_fcnn(nn.Module):
    '''
    2 layer feed forward neural network. 
    Will use leaky ReLU activation functions.
    Activation = {'relu', 'linear','nck','sqgl'}
    '''
    
    def __init__(self, Input_size=3072, Hidden_size=3072, Output_size=1, Activation="relu",
                 alpha=1, beta=0.6, gamma=1, learn=True, scale=1, atten=1, leak=0.01):
        super(simple_fcnn, self).__init__()
        '''
        Inputs: Input_size, Hidden_size, Output_size, Activation
        '''
        # Initialize architecture parameters
        self.Input_size = Input_size
        self.Hidden_size = Hidden_size
        self.Output_size = Output_size
        self.Activation = Activation
        self.learn = learn
        self.scale = scale
        self.atten = atten
        self.leak = leak
        
        # Initialize weights through He initialization (by default in nn.Linear)
        
        self.i2h = nn.Linear(Input_size, Hidden_size, bias=True)
        self.i2h.bias = torch.nn.Parameter(torch.zeros_like(self.i2h.bias))
#         self.i2h.weight = torch.nn.init.normal_(self.i2h.weight, mean=0.0, std=math.sqrt(2/(Input_size)))
        self.i2h.weight = torch.nn.init.kaiming_normal_(self.i2h.weight, a=0.01)

        # Initialize densly connected output layer
        self.h2o = nn.Linear(Hidden_size, Output_size)
        self.h2o.bias = torch.nn.Parameter(torch.zeros_like(self.h2o.bias))
        self.h2o.weight = torch.nn.init.kaiming_normal_(self.h2o.weight, a=0.01)
        
        # Initialize nonlinearities
        self.relu = nn.LeakyReLU(negative_slope=self.leak)
        self.sigmoid = nn.Sigmoid()
        if Activation=='nck':
            self.nck = NCK(alpha, beta, gamma, learn=self.learn, scale=self.scale)
        if Activation=='sqgl':
            self.sqgl = SQGL(alpha, beta, gamma, learn=self.learn, scale=self.scale, atten=self.atten)
        
    def forward(self, x):
        '''
        Forward step for network. Establishes Architecture.
        Inputs: Input
        Outputs: Output
        '''
        # Prepare input for appropriate architecture

        
        # Set Activation function to calculate hidden layer

        if self.Activation == 'relu':
            Hidden = self.relu(self.i2h(x))
        elif self.Activation == 'nck':
            Hidden = self.nck(self.i2h(x))
        elif self.Activation == 'sqgl':
            Hidden = self.sqgl(self.i2h(x))
        else:
            Hidden = self.i2h(x)

        # Calculate Output layer
#         Output = self.sigmoid(self.h2o(Hidden))
        Output = self.h2o(Hidden)
        return(Output)
    
class ktree_gen(nn.Module):
    '''
    k-Tree neural network
    '''
    
    def __init__(self, ds='mnist', Activation="relu", Sparse=True,
                 Input_order=None, Repeats=1, Padded=False,
                 alpha=1, beta=0.6, gamma=1, learn=True, scale=1, atten=1):
        super(ktree_gen, self).__init__()
        '''
        Inputs: ds (dataset), activation, sparse, input_order, repeats, padded
        '''
        # Initialize architecture parameters
        self.ds = ds
        self.Activation = Activation
        self.Sparse = Sparse
        self.Input_order = Input_order
        self.Repeats = Repeats
        self.learn = learn
        self.scale = scale
        self.atten = atten
                
        # Initialize weights
        # Set biases to 0
        # Set kaiming initialize weights with gain to correct for sparsity
        # Set freeze masks
        
        #Specify tree dimensions
        # If using 28x28 datasets...
        if (ds == 'mnist') or (ds == 'fmnist') or (ds == 'kmnist') or (ds == 'emnist'):
            # If padded, use 1024 sized tree, completely binary tree
            if Padded:
                self.k = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            # If not padded, use 784 sized tree, 
            # 7:1 between layers 1 and 2, and layers 2 and 3
            else:
                self.k = [784, 112, 16, 8, 4, 2, 1]
        # If using 3x32x32 datasets...
        elif (ds == 'svhn') or (ds == 'cifar10'):
            # Use 3072 sized tree
            # 3:1 between layers 1 and 2, otherwise binary
            self.k = [3072, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        # If using 16x16 datasets...
        elif ds == 'usps':
            # Use 256 sized tree
            self.k = [256, 128, 64, 32, 16, 8, 4, 2, 1]
        else:
            print('Select a dataset')
            return(None)
        
        # Make layers of tree architecture
        
        # Name each layer in each subtree for reference later
        self.names = np.empty((self.Repeats, len(self.k)-1),dtype=object)
        # Initialize freeze mask for use in training loop
        self.freeze_mask_set = []
        # For each repeat or subtree, make a sparse layer that is initialized correctly
        for j in range(self.Repeats):
            # For each layer within each subtree
            for i in range(len(self.k)-1):
                # Assign name of the layer, indexed by layer (i) and subtree (j)
                name = ''.join(['w',str(j),'_',str(i)])
                # Initialize the layer with the appropriate name
                self.add_module(name, nn.Linear(self.k[i],self.k[i+1]))
                # Set bias of layer to zeros
                self._modules[name].bias = nn.Parameter(torch.zeros_like(self._modules[name].bias)) 
                # Use custom method to re-initialize the layer weights and create freeze mask for that layer
                self._modules[name].weight.data, freeze_mask = self.initialize(self._modules[name])
                # Add the layer name to the list of names
                self.names[j,i] = name
                # Set the freeze mask for the first subtree, which should be the same for all subtrees
                if j < 1:
                    self.freeze_mask_set.append(freeze_mask)
        
        # Initialize root node, aka soma node aka output node
        self.root = nn.Linear(Repeats, 1)
        
        # Initialize nonlinearities
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.nck = NCK(alpha, beta, gamma, learn=self.learn, scale=self.scale)
        self.sqgl = SQGL(alpha, beta, gamma, learn=self.learn, scale=self.scale, atten=self.atten)
    
    def forward(self, x):
        '''
        Forward step for network. Establishes Architecture.
        Inputs: Input
        Outputs: Output
        '''
        
        y_out = []
        # Step through every layer in each subtree of model, applying nonlinearities
        for j in range(self.Repeats):
            y = x
            for i in range(len(self.k)-1):
                if self.Activation == 'relu':
                    y = self.relu(self._modules[self.names[j,i]](y))
                elif self.Activation == 'nck':
                    y = self.nck(self._modules[self.names[j,i]](y))
                elif self.Activation == 'sqgl':
                    y = self.sqgl(self._modules[self.names[j,i]](y))
                else:
                    y = self._modules[self.names[j,i]](y)
            # keep track of pen-ultimate layer outputs
            y_out.append(y)
        
        # Calculate final output, joining the outputs of each subtree together
#         output = self.sigmoid(self.root(torch.cat((y_out), dim=1)))
        output = self.root(torch.cat((y_out), dim=1))

        return(output)
    
    def initialize(self, layer):
        # Kaiming initialize weights accounting for sparsity
        
        # Extract weights from layer we are reinitializing
        weights = layer.weight.data
        # If sparse, change the initializations based on density (sparsity)
        if self.Sparse:
            if weights.shape[1] == 3072: # first layer of 3x32x32 image datasets
                inp_block = torch.ones((1,3))
            elif (weights.shape[1] == 784) or (weights.shape[1] == 112): # first or second layer of 28x28 datasets
                inp_block = torch.ones((1,7))
            else:
                inp_block = torch.ones((1,2)) # all other layers (or 32x32)
            
            # Set up mask for where each node receives a set of inputs of equal size to the input block
            inp_mask = kronecker(torch.eye(weights.shape[0]), inp_block)

            # Calculate density
            density = len(np.where(inp_mask)[0])/len(inp_mask.reshape(-1))

            # Generate Kaiming initialization with gain = 1/density
            weights = torch.nn.init.normal_(weights, mean=0.0, std=math.sqrt(2/(weights.shape[1]*density)))
            
            # Where no inputs will be received, set weights to zero
            weights[inp_mask == 0] = 0
        else: # If not sparse, use typical kaiming normalization
            weights = torch.nn.init.normal_(weights, mean=0.0, std=math.sqrt(2/(weights.shape[1])))
        
        # Generate freeze mask for use in training to keep weights initialized to zero at zero
        mask_gen = torch.zeros_like(weights)
        # Indicate where weights are equal to zero
        freeze_mask = mask_gen == weights
        
        return(weights, freeze_mask)

class synapse_fcnn(nn.Module):
    '''
    2 layer feed forward neural network. 
    Will use leaky ReLU activation functions.
    Activation = {'relu', 'linear','nck','sqgl'}
    '''
    
    def __init__(self, Input_size=3072, Hidden_size=3072, Output_size=1, Activation="relu",
                 alpha=1, beta=0.6, gamma=1, learn=True, scale=1, atten=1, leak=0.01):
        super(synapse_fcnn, self).__init__()
        '''
        Inputs: Input_size, Hidden_size, Output_size, Activation
        '''
        # Initialize architecture parameters
        self.Input_size = Input_size
        self.Hidden_size = Hidden_size
        self.Output_size = Output_size
        self.Activation = Activation
        self.learn = learn
        self.scale = scale
        self.atten = atten
        self.leak = leak
        
        # Initialize Synapse layer
        
        self.syn = Synapse(Input_size)
        
        # Initialize weights through He initialization (by default in nn.Linear)
        
        self.i2h = nn.Linear(Input_size, Hidden_size, bias=True)
        self.i2h.bias = torch.nn.Parameter(torch.zeros_like(self.i2h.bias))
#         self.i2h.weight = torch.nn.init.normal_(self.i2h.weight, mean=0.0, std=math.sqrt(2/(Input_size)))
        self.i2h.weight = torch.nn.init.kaiming_normal_(self.i2h.weight, a=0.01)

        # Initialize densly connected output layer
        self.h2o = nn.Linear(Hidden_size, Output_size)
        self.h2o.bias = torch.nn.Parameter(torch.zeros_like(self.h2o.bias))
        self.h2o.weight = torch.nn.init.kaiming_normal_(self.h2o.weight, a=0.01)
        
        # Initialize nonlinearities
        self.relu = nn.LeakyReLU(negative_slope=self.leak)
        self.sigmoid = nn.Sigmoid()
        self.swish = nn.Hardswish()
        if Activation=='nck':
            self.nck = NCK(alpha, beta, gamma, learn=self.learn, scale=self.scale)
        if Activation=='sqgl':
            self.sqgl = SQGL(alpha, beta, gamma, learn=self.learn, scale=self.scale, atten=self.atten)
        
    def forward(self, x):
        '''
        Forward step for network. Establishes Architecture.
        Inputs: Input
        Outputs: Output
        '''
        # Receive inputs into synapse layer
        
        x = self.syn(x)

        # Set Activation function to calculate hidden layer

        if self.Activation == 'relu':
            Hidden = self.relu(self.i2h(x))
        elif self.Activation == 'nck':
            Hidden = self.nck(self.i2h(x))
        elif self.Activation == 'sqgl':
            Hidden = self.sqgl(self.i2h(x))
        elif self.Activation == 'swish':
            Hidden = self.swish(self.i2h(x))
        else:
            Hidden = self.i2h(x)

        # Calculate Output layer
#         Output = self.sigmoid(self.h2o(Hidden))
        Output = self.h2o(Hidden)
        return(Output)
    
class synapse_ktree_gen(nn.Module):
    '''
    k-Tree neural network
    '''
    
    def __init__(self, ds='mnist', Activation="relu", Sparse=True,
                 Input_order=None, Repeats=1, Padded=False,
                 alpha=1, beta=0.6, gamma=1, learn=True, scale=1, atten=1):
        super(synapse_ktree_gen, self).__init__()
        '''
        Inputs: ds (dataset), activation, sparse, input_order, repeats, padded
        '''
        # Initialize architecture parameters
        self.ds = ds
        self.Activation = Activation
        self.Sparse = Sparse
        self.Input_order = Input_order
        self.Repeats = Repeats
        self.learn = learn
        self.scale = scale
        self.atten = atten
                
        # Initialize weights
        
        # Set biases to 0
        # Set kaiming initialize weights with gain to correct for sparsity
        # Set freeze masks
        
        #Specify tree dimensions
        # If using 28x28 datasets...
        if (ds == 'mnist') or (ds == 'fmnist') or (ds == 'kmnist') or (ds == 'emnist'):
            # If padded, use 1024 sized tree, completely binary tree
            if Padded:
                self.k = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            # If not padded, use 784 sized tree, 
            # 7:1 between layers 1 and 2, and layers 2 and 3
            else:
                self.k = [784, 112, 16, 8, 4, 2, 1]
        # If using 3x32x32 datasets...
        elif (ds == 'svhn') or (ds == 'cifar10'):
            # Use 3072 sized tree
            # 3:1 between layers 1 and 2, otherwise binary
            self.k = [3072, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        # If using 16x16 datasets...
        elif ds == 'usps':
            # Use 256 sized tree
            self.k = [256, 128, 64, 32, 16, 8, 4, 2, 1]
        else:
            print('Select a dataset')
            return(None)
        
        # Make layers of tree architecture
        
        # Name each layer in each subtree for reference later
        self.names = np.empty((self.Repeats, len(self.k)-1),dtype=object)
        self.syn_names = np.empty((self.Repeats), dtype=object)
        # Initialize freeze mask for use in training loop
        self.freeze_mask_set = []
        # For each repeat or subtree, make a sparse layer that is initialized correctly
        for j in range(self.Repeats):
            
            #Initialize synapse layer for each subtree
            syn_name = ''.join(['s',str(j)])
            self.add_module(syn_name, Synapse(self.k[0]))
            self.syn_names[j] = syn_name

            # For each layer within each subtree
            for i in range(len(self.k)-1):
                # Assign name of the layer, indexed by layer (i) and subtree (j)
                name = ''.join(['w',str(j),'_',str(i)])
                # Initialize the layer with the appropriate name
                self.add_module(name, nn.Linear(self.k[i],self.k[i+1]))
                # Set bias of layer to zeros
                self._modules[name].bias = nn.Parameter(torch.zeros_like(self._modules[name].bias)) 
                # Use custom method to re-initialize the layer weights and create freeze mask for that layer
                self._modules[name].weight.data, freeze_mask = self.initialize(self._modules[name])
                # Add the layer name to the list of names
                self.names[j,i] = name
                # Set the freeze mask for the first subtree, which should be the same for all subtrees
                if j < 1:
                    self.freeze_mask_set.append(freeze_mask)
                    
                
        
        # Initialize root node, aka soma node aka output node
        self.root = nn.Linear(Repeats, 1)
        
        # Initialize nonlinearities
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.nck = NCK(alpha, beta, gamma, learn=self.learn, scale=self.scale)
        self.sqgl = SQGL(alpha, beta, gamma, learn=self.learn, scale=self.scale, atten=self.atten)
    
    def forward(self, x):
        '''
        Forward step for network. Establishes Architecture.
        Inputs: Input
        Outputs: Output
        '''
        
        y_out = []
        # Step through every layer in each subtree of model, applying nonlinearities
        for j in range(self.Repeats):
            y = self._modules[self.syn_names[j]](x) # Synapse layer for each subtree
            for i in range(len(self.k)-1):
                if self.Activation == 'relu':
                    y = self.relu(self._modules[self.names[j,i]](y))
                elif self.Activation == 'nck':
                    y = self.nck(self._modules[self.names[j,i]](y))
                elif self.Activation == 'sqgl':
                    y = self.sqgl(self._modules[self.names[j,i]](y))
                else:
                    y = self._modules[self.names[j,i]](y)
            # keep track of pen-ultimate layer outputs
            y_out.append(y)
        
        # Calculate final output, joining the outputs of each subtree together
#         output = self.sigmoid(self.root(torch.cat((y_out), dim=1)))
        output = self.root(torch.cat((y_out), dim=1))

        return(output)
    
    def initialize(self, layer):
        # Kaiming initialize weights accounting for sparsity
        
        # Extract weights from layer we are reinitializing
        weights = layer.weight.data
        # If sparse, change the initializations based on density (sparsity)
        if self.Sparse:
            if weights.shape[1] == 3072: # first layer of 3x32x32 image datasets
                inp_block = torch.ones((1,3))
            elif (weights.shape[1] == 784) or (weights.shape[1] == 112): # first or second layer of 28x28 datasets
                inp_block = torch.ones((1,7))
            else:
                inp_block = torch.ones((1,2)) # all other layers (or 32x32)
            
            # Set up mask for where each node receives a set of inputs of equal size to the input block
            inp_mask = kronecker(torch.eye(weights.shape[0]), inp_block)

            # Calculate density
            density = len(np.where(inp_mask)[0])/len(inp_mask.reshape(-1))

            # Generate Kaiming initialization with gain = 1/density
            weights = torch.nn.init.normal_(weights, mean=0.0, std=math.sqrt(2/(weights.shape[1]*density)))
            
            # Where no inputs will be received, set weights to zero
            weights[inp_mask == 0] = 0
        else: # If not sparse, use typical kaiming normalization
            weights = torch.nn.init.normal_(weights, mean=0.0, std=math.sqrt(2/(weights.shape[1])))
        
        # Generate freeze mask for use in training to keep weights initialized to zero at zero
        mask_gen = torch.zeros_like(weights)
        # Indicate where weights are equal to zero
        freeze_mask = mask_gen == weights
        
        return(weights, freeze_mask)
    
class ktree_sparse(nn.Module):
    '''
    k-Tree neural network
    '''
    
    def __init__(self, ds='mnist', Activation="relu", 
                 Input_order=None, Repeats=1, Padded=True,
                 alpha=1, beta=1, gamma=1, learn=True, scale=1, atten=1,
                 synapse=True, leak=0.01, Node_vary=True, positive=False):
        super(ktree_sparse, self).__init__()
        '''
        Inputs: ds (dataset), activation, sparse, input_order, repeats, padded
        '''
        # Initialize architecture parameters
        self.ds = ds
        self.Activation = Activation
        self.Input_order = Input_order
        self.Repeats = Repeats
        self.learn = learn
        self.scale = scale
        self.atten = atten
        self.synapse = synapse
        self.leak = leak
        self.Node_vary = Node_vary
        self.positive = positive
                
        # Initialize weights
        # Set biases to 0
        # Set kaiming initialize weights with gain to correct for sparsity
        # Set freeze masks
        
        #Specify tree dimensions
        # If using 28x28 datasets...
        if (ds == 'mnist') or (ds == 'fmnist') or (ds == 'kmnist') or (ds == 'emnist'):
            # If padded, use 1024 sized tree, completely binary tree
            if Padded:
                self.k = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            # If not padded, use 784 sized tree, 
            # 7:1 between layers 1 and 2, and layers 2 and 3
            else:
                self.k = [784, 112, 16, 8, 4, 2, 1]
        # If using 3x32x32 datasets...
        elif (ds == 'svhn') or (ds == 'cifar10'):
            # Use 3072 sized tree
            # 3:1 between layers 1 and 2, otherwise binary
            self.k = [3072, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        # If using 16x16 datasets...
        elif ds == 'usps':
            # Use 256 sized tree
            self.k = [256, 128, 64, 32, 16, 8, 4, 2, 1]
        else:
            print('Select a dataset')
            return(None)
        
        # Make layers of tree architecture
        
        # Name each layer in each subtree for reference later
        self.names = np.empty((self.Repeats, len(self.k)-1),dtype=object)
        if self.synapse:
            self.syn_names = np.empty((self.Repeats), dtype=object)
            
        if self.Node_vary:
            self.sqgl_names = np.empty((self.Repeats, len(self.k)-1),dtype=object)

        # Initialize freeze mask for use in training loop
        self.freeze_mask_set = []
        # For each repeat or subtree, make a sparse layer that is initialized correctly
        for j in range(self.Repeats):
            
            if self.synapse:
                #Initialize synapse layer for each subtree
                syn_name = ''.join(['s',str(j)])
                self.add_module(syn_name, Synapse(self.k[0]))
                self.syn_names[j] = syn_name
                
            if self.Node_vary == True:
                for i in range(len(self.k)-1):
                    # Assign name of each sqgl layer, indexed by layer (i) and subtree (j)
                    sqgl_name = ''.join(['sq',str(j),'_',str(i)])
                    # Initialize the layer with the appropriate name
                    self.add_module(sqgl_name, slc.SparseLinear(3*self.k[i+1],
                                                                self.k[i+1],
                                                                connectivity=self.sqgl_connectivity(self.k[i+1]),
                                                                bias=False,
                                                                sqgl_true=True))
                    # Add activation layer name to list of names
                    self.sqgl_names[j,i] = sqgl_name
            
            # For each layer within each subtree
            for i in range(len(self.k)-1):
                # Assign name of the layer, indexed by layer (i) and subtree (j)
                name = ''.join(['w',str(j),'_',str(i)])
                # Initialize the layer with the appropriate name
                self.add_module(name, slc.SparseLinear(self.k[i], 
                                                      self.k[i+1], 
                                                      connectivity=self.layer_connectivity(self.k[i]),
                                                      bias=True,
                                                      positive=self.positive))
                # Add the layer name to the list of names
                self.names[j,i] = name
                self._modules[self.names[j,i]].bias.data.zero_()
        
        # Initialize root node, aka soma node aka output node
        self.root = nn.Linear(Repeats, 1, bias=True)
        self.root.bias.data.zero_()
        
        # Initialize nonlinearities
        self.relu = nn.LeakyReLU(negative_slope=leak)
        self.sigmoid = nn.Sigmoid()
        self.swish = nn.Hardswish()
        self.nck = NCK(alpha, beta, gamma, learn=self.learn, scale=self.scale)
        self.sqgl = SQGL(alpha, beta, gamma, learn=self.learn, scale=self.scale, atten=self.atten)
        
        if self.Node_vary == True:
            self.f_na = F_Na(self.scale)
            self.f_ca = F_Ca(self.scale)
            self.f_k = F_K(self.scale)
    
    def forward(self, x):
        '''
        Forward step for network. Establishes Architecture.
        Inputs: Input
        Outputs: Output
        '''
        
        y_out = []
        # Step through every layer in each subtree of model, applying nonlinearities
        for j in range(self.Repeats):
            if self.synapse:
                y = self._modules[self.syn_names[j]](x) # Synapse layer for each subtree
            else:
                y = x
                
            for i in range(len(self.k)-1):
                if self.Activation == 'relu':
                    y = self.relu(self._modules[self.names[j,i]](y))
                elif self.Activation == 'nck':
                    y = self.nck(self._modules[self.names[j,i]](y))
                elif self.Activation == 'sqgl':
                    if self.Node_vary == True: # If varying sqgl by node
                        # First put through linear layer
                        y = self._modules[self.names[j,i]](y)
                        # Then do step one of sqgl nonlinearity
                        interm_act = torch.cat((self.f_na(y), self.f_ca(y), self.f_k(y)), axis=1)
                        # Weighted sum of nonlinearities, added to linear component
                        # Then multiplied by an attenuation factor
                        y = self.atten*(y + self._modules[self.sqgl_names[j,i]](interm_act))
                    else:
                        y = self.sqgl(self._modules[self.names[j,i]](y))
                elif self.Activation == 'sigmoid':
                    y = self.sigmoid(self._modules[self.names[j,i]](y))
                elif self.Activation == 'silu':
                    y = self.silu(self._modules[self.names[j,i]](y))
                elif self.Activation == 'swish':
                    y = self.swish(self._modules[self.names[j,i]](y))
                else:
                    y = self._modules[self.names[j,i]](y)
            # keep track of pen-ultimate layer outputs
            y_out.append(y)
            
        # Calculate final output, joining the outputs of each subtree together
#         output = self.sigmoid(self.root(torch.cat((y_out), dim=1)))
        output = self.root(torch.cat((y_out), dim=1))

        return(output)
    
    def layer_connectivity(self, in_features):
        if in_features == 3072:
            inp_block = torch.ones((1,3))
        elif (in_features == 784) or (in_features == 112): # first or second layer of 28x28 datasets
            inp_block = torch.ones((1,7))
        else:
            inp_block = torch.ones((1,2)) # all other layers (or 32x32)

        inp_mask = kronecker(torch.eye(int(in_features/inp_block.size()[1])), inp_block)

        ix = inp_mask.nonzero(as_tuple=False)
        return(ix.t())
    
    def sqgl_connectivity(self, in_features):   
        # Only works as an activation function layer if input to this layer is 1x1x3N where N is original input size
        inp_block = torch.eye(in_features)
        inp_mask = torch.cat((inp_block, inp_block, inp_block), axis=0)
        inp_mask = inp_mask.t()
        ix = inp_mask.nonzero(as_tuple=False)
        return(ix.t())
    
class asym_tree_gen(nn.Module):
    '''
    asym-Tree neural network
    '''
    
    def __init__(self, ds='mnist', Activation="relu", 
                 Input_order=None, Repeats=1, Padded=True,
                 alpha=1, beta=0.6, gamma=1, learn=True, scale=1, atten=1,
                 synapse=False, tree=None, leak=0.01):
        super(asym_tree_gen, self).__init__()
        '''
        Inputs: ds (dataset), activation, sparse, input_order, repeats, padded
        '''
        # Initialize architecture parameters
        self.ds = ds
        self.Activation = Activation
        self.Input_order = Input_order
        self.Repeats = Repeats
        self.learn = learn
        self.scale = scale
        self.atten = atten
        self.Synapse = synapse
        self.tree = tree
        self.num_leaves = len(self.find_all_leaf_ids(self.tree))
        self.leak = leak
        
        
        if self.tree is None:
            raise TypeError('Did not specify tree')
                
        # Initialize weights
        # Set biases to 0
        # Set kaiming initialize weights with gain to correct for sparsity
        # Set freeze masks
        
        #Specify tree dimensions
        # If using 28x28 datasets...
        if (ds == 'mnist') or (ds == 'fmnist') or (ds == 'kmnist') or (ds == 'emnist'):
            # If padded, use 1024 sized tree, completely binary tree
            if Padded:
                self.input_size = 1024
            # If not padded, use 784 sized tree, 
            else:
                self.input_size = 784
        # If using 3x32x32 datasets...
        elif (ds == 'svhn') or (ds == 'cifar10'):
            # Use 3072 sized tree
            self.input_size = 3072
        # If using 16x16 datasets...
        elif ds == 'usps':
            # Use 256 sized tree
            self.input_size = 256
        else:
            print('Select a dataset')
            return(None)
        
        # Make layers of tree architecture
        
        # Make list of connectivity matrices for purpose of making sparse layers
        self.connectivity_matrices = self.seq_adj_mat(self.tree)
        
        # Name each layer in each subtree for reference later
        self.names = np.empty((self.Repeats, len(self.connectivity_matrices)),dtype=object)
        if self.Synapse:
            self.syn_names = np.empty((self.Repeats), dtype=object)

        # For each repeat or subtree, make a sparse layer that is initialized correctly
        for j in range(self.Repeats):
            
            if self.Synapse:
                #Initialize synapse layer for each subtree
                syn_name = ''.join(['s',str(j)])
                self.add_module(syn_name, Synapse(self.input_size))
                self.syn_names[j] = syn_name
            
            # For each layer within each subtree
            for i, connectivity_matrix in enumerate(self.connectivity_matrices):
                # Assign name of the layer, indexed by layer (i) and subtree (j)
                name = ''.join(['w',str(j),'_',str(i)])
                # Initialize the layer with the appropriate name
                self.add_module(name, slc.SparseLinear(self.num_leaves + len(self.tree), 
                                                       len(self.tree),
                                                       connectivity=connectivity_matrix,
                                                       bias=True))
#                 print(i,connectivity_matrix)
#                 print(self._modules[name])
                # Add the layer name to the list of names
                self.names[j,i] = name
                self._modules[self.names[j,i]].bias.data.zero_()
        
        # Initialize root node, aka soma node aka output node
        self.root = nn.Linear(Repeats, 1, bias=True)
        self.root.bias.data.zero_()
        
        # Initialize nonlinearities
        self.relu = nn.LeakyReLU(negative_slope=self.leak)
        self.sigmoid = nn.Sigmoid()
        self.nck = NCK(alpha, beta, gamma, learn=self.learn, scale=self.scale)
        self.sqgl = SQGL(alpha, beta, gamma, learn=self.learn, scale=self.scale, atten=self.atten)
    
    def find_all_leaf_ids(self,tree):
        """
        just see which nodes are not in the list
        like in tree = [-1,0,0,1,1,2,2]
        leaves are [3, 4, 5, 6]
        """
        all_leaf_ids = [i for i in range(len(tree)) if i not in tree and tree[i] is not None]
        return(np.array(all_leaf_ids))

    def find_all_branch_ids(self,tree):
        '''
        just see which nodes are not in the list
        like in tree = [-1,0,0,1,1,2,2]
        leaves are [3, 4, 5, 6]
        '''
        all_branch_ids = [i for i in range(len(tree)) if i in tree and tree[i] is not None]
        return(np.array(all_branch_ids))
    
    def path_lengths(self, tree):
        paths = np.zeros(len(tree))

        for i in range(len(tree)):
            node = i
            branch = node
            path_length = 0
            
            while branch != 0:
                branch = tree[node]
                node = branch
                path_length += 1
            paths[i] = path_length

        return(paths.astype(int))


    def seq_adj_mat(self, tree):
        paths = self.path_lengths(tree)
        leaves = self.find_all_leaf_ids(tree)
        branches = self.find_all_branch_ids(tree)
        ids = np.arange(len(tree))
        i = 1
        j = 1

        adj_mats = []

        for path_len in reversed(range(max(paths)+1)):
            adj_mat = np.zeros((len(tree), len(tree) + self.num_leaves))
            idx = np.where(path_len == paths, True, False)
            nodes = ids[idx]
            leaf_nodes = list(filter(lambda x: x in leaves, nodes))
            branch_nodes = list(filter(lambda x: x in branches, nodes))
            for leaf in leaf_nodes:
                adj_mat[leaf, len(leaves) - i] = 1
                i += 1
            for branch in branch_nodes:
                adj_mat[branch, len(tree) + len(leaves) - j] = 1
                j += 1
                adj_mat[branch, len(tree) + len(leaves) - j] = 1
                j += 1
            # Change adj_mats into sparse format
            sparse_idx = []
            idxs = np.where(adj_mat)
            for idx in idxs:
                sparse_idx.append(list(idx))
            sparse_idx = torch.LongTensor(sparse_idx)
            adj_mats.append(sparse_idx)
        return(adj_mats)

    def forward(self, x):
        '''
        Forward step for network. Establishes Architecture.
        Inputs: Input
        Outputs: Output
        '''
        
        y_out = []
        # Step through every layer in each subtree of model, applying nonlinearities
        for j in range(self.Repeats):
            y = x.clone()
#             print(self.num_leaves)
            if y.shape[1] < self.num_leaves:
                filler = torch.zeros((y.shape[0], self.num_leaves)).cuda()
                filler[:,:y.shape[1]] = y
                y = filler
            elif y.shape[1] > self.num_leaves:
                y = y[:, :self.num_leaves]
            hidden = torch.cat((y, torch.zeros(y.shape[0], len(self.tree)).cuda()), dim=1)
            for i in range(len(self.connectivity_matrices)):
#                 print(i,j)
                if self.Activation == 'relu':
                    hidden = self.relu(self._modules[self.names[j,i]](hidden))
                    hidden = torch.cat((y, hidden), dim=1)
                elif self.Activation == 'nck':
                    hidden = self.nck(self._modules[self.names[j,i]](hidden))
                    hidden = torch.cat((y, hidden), dim=1) 
                elif self.Activation == 'sqgl':
                    hidden = self.sqgl(self._modules[self.names[j,i]](hidden))
                    hidden = torch.cat((y, hidden), dim=1) 
                else:
                    hidden = self._modules[self.names[j,i]](hidden)
                    hidden = torch.cat((y, hidden), dim=1) 
#             # keep track of pen-ultimate layer outputs
            y_out.append(hidden[:,self.num_leaves].reshape(y.shape[0],1))
#         print('yout',y_out[0].shape)
            
        # Calculate final output, joining the outputs of each subtree together
#         output = self.sigmoid(self.root(torch.cat((y_out), dim=1)))
        output = self.root(torch.cat((y_out), dim=1))
#         output = y
    
        return(output)
    
def prepare_tree(n):
    return(np.array(n[0][:,-1]).astype(int)-1)

def find_all_leaf_ids(tree):
    """
    just see which nodes are not in the list
    like in tree = [-1,0,0,1,1,2,2]
    leaves are [3, 4, 5, 6]
    """
    all_leaf_ids = [i for i in range(len(tree)) if i not in tree and tree[i] is not None]
    return(np.array(all_leaf_ids))

def find_target_tree(mcn_trees):
    leaves = np.zeros((len(mcn_trees),2))
    for i in range(len(mcn_trees)):
        n = mcn_trees[i]
        tree = (n[0][:,-1]).astype(int) -1
        leaves[i,1] = len(find_all_leaf_ids(tree))
        leaves[i,0] = i
    num_leafs = leaves[0,1]
    if num_leafs > 14**2 and num_leafs < 18**2 :
        targ_diff = 256
    if num_leafs > 26**2 and num_leafs < 34**2 :
        targ_diff = 1024
    if num_leafs > 3*30**2 and num_leafs < 3*34**2 :
        targ_diff = 3072
    leaves = np.concatenate((leaves,abs(leaves[:,1]-targ_diff).reshape(-1,1)),1)
    # print(leaves)
    target = np.argmin(leaves[:,2])
    print('Target Tree:', leaves[target,:])
    return(target)


class ktree_synapse(nn.Module):
    '''
    k-Tree neural network
    '''
    
    def __init__(self, ds='mnist', Activation="relu", 
                 Input_order=None, Repeats=1, Padded=True,
                 alpha=1, beta=1, gamma=1, learn=True, scale=1, atten=1,
                 leak=0.01, Node_vary=True, positive=False):
        super(ktree_synapse, self).__init__()
        '''
        Inputs: ds (dataset), activation, sparse, input_order, repeats, padded
        '''
        # Initialize architecture parameters
        self.ds = ds
        self.Activation = Activation
        self.Input_order = Input_order
        self.Repeats = Repeats
        self.learn = learn
        self.scale = scale
        self.atten = atten
        self.leak = leak
        self.Node_vary = Node_vary
        self.positive = positive
                
        # Initialize weights
        # Set biases to 0
        # Set kaiming initialize weights with gain to correct for sparsity
        # Set freeze masks
        
        #Specify tree dimensions
        # If using 28x28 datasets...
        if (ds == 'mnist') or (ds == 'fmnist') or (ds == 'kmnist') or (ds == 'emnist'):
            # If padded, use 1024 sized tree, completely binary tree
            if Padded:
                self.k = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            # If not padded, use 784 sized tree, 
            # 7:1 between layers 1 and 2, and layers 2 and 3
            else:
                self.k = [784, 112, 16, 8, 4, 2, 1]
        # If using 3x32x32 datasets...
        elif (ds == 'svhn') or (ds == 'cifar10'):
            # Use 3072 sized tree
            # 3:1 between layers 1 and 2, otherwise binary
            self.k = [3072, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        # If using 16x16 datasets...
        elif ds == 'usps':
            # Use 256 sized tree
            self.k = [256, 128, 64, 32, 16, 8, 4, 2, 1]
        else:
            print('Select a dataset')
            return(None)
        
        # Make layers of tree architecture
        
        # Name each layer in each subtree for reference later
        self.names = np.empty((self.Repeats, len(self.k)-1),dtype=object)
        self.syn_names = np.empty((self.Repeats), dtype=object)
            
        if self.Node_vary:
            self.sqgl_names = np.empty((self.Repeats, len(self.k)-1),dtype=object)

        # Initialize freeze mask for use in training loop
        self.freeze_mask_set = []
        # For each repeat or subtree, make a sparse layer that is initialized correctly
        for j in range(self.Repeats):
            
            #Initialize synapse layer for each subtree
            syn_name = ''.join(['s',str(j)])
            self.add_module(syn_name, Synapse(self.k[0]))
            self.syn_names[j] = syn_name
                
            if self.Node_vary == True:
                for i in range(len(self.k)-1):
                    # Assign name of each sqgl layer, indexed by layer (i) and subtree (j)
                    sqgl_name = ''.join(['sq',str(j),'_',str(i)])
                    # Initialize the layer with the appropriate name
                    self.add_module(sqgl_name, slc.SparseLinear(3*self.k[i+1],
                                                                self.k[i+1],
                                                                connectivity=self.sqgl_connectivity(self.k[i+1]),
                                                                bias=False,
                                                                sqgl_true=True))
                    # Add activation layer name to list of names
                    self.sqgl_names[j,i] = sqgl_name
            
            # For each layer within each subtree
            for i in range(len(self.k)-1):
                # Assign name of the layer, indexed by layer (i) and subtree (j)
                name = ''.join(['w',str(j),'_',str(i)])
                # Initialize the layer with the appropriate name
                self.add_module(name, slc.SparseLinear(self.k[i], 
                                                      self.k[i+1], 
                                                      connectivity=self.layer_connectivity(self.k[i]),
                                                      bias=True,
                                                      positive=self.positive))
                # Add the layer name to the list of names
                self.names[j,i] = name
                self._modules[self.names[j,i]].bias.data.zero_()
        
        # Initialize root node, aka soma node aka output node
        self.root = nn.Linear(Repeats, 1, bias=True)
        self.root.bias.data.zero_()
        
        # Initialize nonlinearities
        self.relu = nn.LeakyReLU(negative_slope=leak)
        self.sigmoid = nn.Sigmoid()
        self.swish = nn.Hardswish()
        self.nck = NCK(alpha, beta, gamma, learn=self.learn, scale=self.scale)
        
        if self.Node_vary == True:
            self.f_na = F_Na(self.scale)
            self.f_ca = F_Ca(self.scale)
            self.f_k = F_K(self.scale)
        else:
            self.sqgl = SQGL(alpha, beta, gamma, learn=self.learn, scale=self.scale, atten=self.atten)

    
    def forward(self, x):
        '''
        Forward step for network. Establishes Architecture.
        Inputs: Input
        Outputs: Output
        '''
        
        y_out = []
        loss = 0
        hinge_loss = Hinge_loss(margin=0.5)
        # Step through every layer in each subtree of model, applying nonlinearities
        for j in range(self.Repeats):
            y = self._modules[self.syn_names[j]](x) # Synapse layer for each subtree
                
            for i in range(len(self.k)-1):
                if self.Activation == 'relu':
                    y = self.relu(self._modules[self.names[j,i]](y))
                    loss += self.hinge_criterion(y)
                elif self.Activation == 'nck':
                    y = self.nck(self._modules[self.names[j,i]](y))
                    loss += self.hinge_criterion(y)
                elif self.Activation == 'sqgl':
                    if self.Node_vary == True: # If varying sqgl by node
                        # First put through linear layer
                        y = self._modules[self.names[j,i]](y)
                        # Then do step one of sqgl nonlinearity
                        interm_act = torch.cat((self.f_na(y), self.f_ca(y), self.f_k(y)), axis=1)
                        # Weighted sum of nonlinearities, added to linear component
                        # Then multiplied by an attenuation factor
                        y = self.atten*(y + self._modules[self.sqgl_names[j,i]](interm_act))
                        #Calculate loss
                        loss += self.hinge_criterion(y)
                    else:
                        y = self.sqgl(self._modules[self.names[j,i]](y))
                        loss += self.hinge_criterion(y)
                elif self.Activation == 'sigmoid':
                    y = self.sigmoid(self._modules[self.names[j,i]](y))
                    loss += self.hinge_criterion(y)
                elif self.Activation == 'silu':
                    y = self.silu(self._modules[self.names[j,i]](y))
                    loss += self.hinge_criterion(y)
                elif self.Activation == 'swish':
                    y = self.swish(self._modules[self.names[j,i]](y))
                    loss += self.hinge_criterion(y)
                else:
                    y = self._modules[self.names[j,i]](y)
                    loss += self.hinge_criterion(y)
            # keep track of pen-ultimate layer outputs
            y_out.append(y)
            
        # Calculate final output, joining the outputs of each subtree together
#         output = self.sigmoid(self.root(torch.cat((y_out), dim=1)))
        output = self.root(torch.cat((y_out), dim=1))

        return(output, loss)
    
    def layer_connectivity(self, in_features):
        if in_features == 3072:
            inp_block = torch.ones((1,3))
        elif (in_features == 784) or (in_features == 112): # first or second layer of 28x28 datasets
            inp_block = torch.ones((1,7))
        else:
            inp_block = torch.ones((1,2)) # all other layers (or 32x32)

        inp_mask = kronecker(torch.eye(int(in_features/inp_block.size()[1])), inp_block)

        ix = inp_mask.nonzero(as_tuple=False)
        return(ix.t())
    
    def sqgl_connectivity(self, in_features):   
        # Only works as an activation function layer if input to this layer is 1x1x3N where N is original input size
        inp_block = torch.eye(in_features)
        inp_mask = torch.cat((inp_block, inp_block, inp_block), axis=0)
        inp_mask = inp_mask.t()
        ix = inp_mask.nonzero(as_tuple=False)
        return(ix.t())
    
    def hinge_criterion(self, activity):
        # Specify targets with same size as layer
        hinge_loss = Hinge_loss(margin=0.5)
        target = - torch.ones_like(activity)
        # Loss = hinge(v-vmax) + hinge(vmin-v)
        return(hinge_loss(activity-51, target) + hinge_loss(-71-activity, target))