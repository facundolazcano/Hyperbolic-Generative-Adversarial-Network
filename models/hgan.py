import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
#import hyptorch.nn as hypnn
import models.HypLayers as hyply

## modes to train c
noTrainC = 0
trainC = 1
trainC_exp = 2
trainC_per_layer = 3

# predefinited node sizes for dataset
nodeSizesDiscriminatorMNIST = [784, 1024, 512, 256, 1]
#version de capas mas chicas para probar hipotesis de que la red sirve aunque sea menor
#nodeSizesDiscriminatorMNIST = [784, 512, 256, 128, 1]
nodeSizesGeneratorMNIST = [128, 256, 512, 1024, 784]
nodeSizesDiscriminatorTFD48 = [2304, 4096,2048, 1024, 1]
nodeSizesGeneratorTFD48 = [512, 1024, 2048, 4096, 2304]
nodeSizesDiscriminatorCIFAR10 = [3072, 4096,2048, 1024, 1]
nodeSizesGeneratorCIFAR10 = [128, 1024, 2048, 4096, 3072]

##test
nodeSizesDiscriminatorMNIST_2 = [784, 1024, 512, 256, 128, 3]

# Makers
def maker_hyp_nnet(**kwargs):#cfg, c, nodeSizes, bias=False, modeTrainC=noTrainC, activation=None):
    cfg = kwargs['cfg']
    c = kwargs['c']
    nodeSizes = kwargs['nodeSizes']
    bias = kwargs['bias']
    modeTrainC = kwargs['modeTrainC']
    
    layers = nn.ModuleList()
    j = 0
    for i in cfg:
        # to poincaré
        if i == 't':
            #if modeTrainC == noTrainC:
            layers.append(hyply.ToPoincare(c=c,
                                           train_x=False,
                                           train_c=False))
        elif i == 'h':
            layers.append(hyply.HypLinear(nodeSizes[j],
                                               nodeSizes[j + 1],
                                               c=c,
                                               bias=bias))
            j += 1
        # euclidean layer
        elif i == 'e':
            layers.append(nn.Linear(nodeSizes[j],
                                    nodeSizes[j + 1],
                                    bias=bias))
            j += 1
        elif i == 'f':
            #if modeTrainC == noTrainC:
            layers.append(hyply.FromPoincare(c=c,
                                             train_x=False,
                                             train_c=False))
        elif i == 'm':
            layers.append(hyply.MixHypEqualLinear(nodeSizes[j],
                                                  nodeSizes[j + 1],
                                                  c=c,
                                                  activation='None'))
            j += 1
                                                  
            #else:
            #    layers.append(hypnn.FromPoincare(c=c,
            #                                     train_x=False,
            #                                     train_c=True))
        else:
            raise ValueError('option not implemented')
    return layers

# Discriminator Net
class HypDiscriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    _defaults = {
        "c": 0.01,
        "modeTrainC": noTrainC,
        "nodeSizes": [2304, 4096, 2048, 1024, 1],
        "cfg": 'eeee',
        "lm_c": 0.01, #0.00001,
        "epsilon": 1e-6
    }


    def __init__(self, **kwargs):
        super(HypDiscriminator, self).__init__()

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.layers = maker_hyp_nnet(**kwargs)
        
        if self.modeTrainC == trainC:
            self.c = self.c / self.lm_c
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon = torch.as_tensor(self.epsilon)
            self.lm_c = torch.as_tensor(self.lm_c)
            
        elif self.modeTrainC == trainC_exp:
            self.c = np.log(self.c / self.lm_c)
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon = torch.as_tensor(self.epsilon)
            self.lm_c = torch.as_tensor(self.lm_c)

    def forward(self, x):
        
        if self.modeTrainC == trainC:
            c = torch.abs(self.c * self.lm_c) + self.epsilon
        elif self.modeTrainC == trainC_exp:
            c = torch.exp(self.c * self.lm_c) + self.epsilon

        else:
            c = self.c

        layers = list(enumerate(self.cfg))
        for i, j in layers[:-1]:
            if j == 'h' or j == 'm':
                x = self.layers[i](x, c)
                x = F.leaky_relu(x, 0.2)
                x = F.dropout(x, 0.1)            
                
            elif j == 'e':
                x = self.layers[i](x)
                x = F.leaky_relu(x, 0.2)
                x = F.dropout(x, 0.1)

            else:
                x = self.layers[i](x, c)

        # end layer
        i, j = layers[-1]
        if j == 'e':
            x = self.layers[i](x)
        else:
            x = self.layers[i](x, c)
        return x
    
    def get_real_c(self):
        
        if self.modeTrainC == noTrainC:
            return self.c
        elif self.modeTrainC == trainC:
            return self.c.item() * self.lm_c.item() + self.epsilon.item()
        elif self.modeTrainC == trainC_exp:
            return np.exp(self.c.item() * self.lm_c.item()) + self.epsilon.item()
        
        

## Generator
class HypGenerator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    _defaults = {
        "c": 0.01,
        "modeTrainC": noTrainC,
        "nodeSizes": [128, 1024, 2048, 4096, 2304],
        "cfg": 'eeee',
        "lm_c": 0.01, #0.00001,
        "epsilon": 1e-6
    }

    def __init__(self, **kwargs):
        super(HypGenerator, self).__init__()

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.layers = maker_hyp_nnet(**kwargs)
        
        if self.modeTrainC == trainC:
            self.c = self.c / self.lm_c
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon = torch.as_tensor(self.epsilon)
            self.lm_c = torch.as_tensor(self.lm_c)
        elif self.modeTrainC == trainC_exp:
            self.c = np.log(self.c)/ self.lm_c
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon = torch.as_tensor(self.epsilon)
            self.lm_c = torch.as_tensor(self.lm_c)

    def forward(self, x):

        layers = list(enumerate(self.cfg))
        
        if self.modeTrainC == trainC:
            c = torch.abs(self.c * self.lm_c) + self.epsilon 
        elif self.modeTrainC == trainC_exp:
            c = torch.exp(self.c * self.lm_c) + self.epsilon
        else:
            c = self.c


        for i, j in layers[:-1]:
            if j == 'h' or j == 'm':
                x = self.layers[i](x, c)
                x = F.leaky_relu(x, 0.2)
                
            elif j == 'e':
                x = self.layers[i](x)
                x = F.leaky_relu(x, 0.2)
            else:
                x = self.layers[i](x, c)

        # end layer
        i, j = layers[-1]
        if j == 'e':
            x = self.layers[i](x)
        else:
            x = self.layers[i](x, c)

        return F.tanh(x)
    
    def get_real_c(self):

        if self.modeTrainC == noTrainC:
            return self.c
        elif self.modeTrainC == trainC:
            return np.abs(self.c.item() * self.lm_c.item()) + self.epsilon.item()
        elif self.modeTrainC == trainC_exp:
            return np.exp(self.c.item() * self.lm_c.item()) + self.epsilon.item()


# Discriminator Net
class HypConditionalDiscriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    _defaults = {
        "c": 0.01,
        "modeTrainC": noTrainC,
        "nodeSizes": [2304, 4096, 2048, 1024, 1],
        "cfg": 'thhhh',
        "n_classes": 10,
        "lm_c": 0.01, #0.00001,
        "epsilon": 1e-6
    }

    def __init__(self, **kwargs):
        super(HypConditionalDiscriminator, self).__init__()

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)
        self.nodeSizes[0] += self.n_classes

        self.layers = maker_hyp_nnet(**kwargs)
        
        # set training C
        if self.modeTrainC == trainC:
            self.c = self.c / self.lm_c
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon = torch.as_tensor(self.epsilon)
            self.lm_c = torch.as_tensor(self.lm_c)
        elif self.modeTrainC == trainC_exp:
            self.c = np.log(self.c)/ self.lm_c
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon = torch.as_tensor(self.epsilon)
            self.lm_c = torch.as_tensor(self.lm_c)

    def forward(self, x, y):
        
        if self.modeTrainC == trainC:
            c = torch.abs(self.c * self.lm_c) + self.epsilon 
        elif self.modeTrainC == trainC_exp:
            c = torch.exp(self.c * self.lm_c) + self.epsilon
        else:
            c = self.c

        x = torch.cat((self.label_emb(y), x), -1)

        layers = list(enumerate(self.cfg))
        for i, j in layers[:-1]:
            if j == 'h':
                x = self.layers[i](x, c)
                x = F.leaky_relu(x, 0.2)
                x = F.dropout(x, 0.1)
            elif j == 'e':
                x = self.layers[i](x)
                x = F.leaky_relu(x, 0.2)
                x = F.dropout(x, 0.1)
            else:
                x = self.layers[i](x, c)
                
        # end layer
        # end layer
        i, j = layers[-1]
        if j == 'e':
            x = self.layers[i](x)
        else:
            x = self.layers[i](x, c)
        return x
    
    def get_real_c(self):
        
        if self.modeTrainC == noTrainC:
            return self.c
        elif self.modeTrainC == trainC:
            return self.c.item() * self.lm_c.item() + self.epsilon.item()
        elif self.modeTrainC == trainC_exp:
            return np.exp(self.c.item() * self.lm_c.item()) + self.epsilon.item()

## Generator
class HypConditionalGenerator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    _defaults = {
        "c": 0.01,
        "modeTrainC": noTrainC,
        "nodeSizes": [128, 1024, 2048, 4096, 2304],
        "cfg": 'eeee',
        "n_classes": 10,
        "lm_c": 0.01, #0.00001,
        "epsilon": 1e-6
    }

    def __init__(self, **kwargs):
        super(HypConditionalGenerator, self).__init__()

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)
        self.nodeSizes[0] += self.n_classes

        self.layers = maker_hyp_nnet(**kwargs)
                
        # set training C
        if self.modeTrainC == trainC:
            self.c = self.c / self.lm_c
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon = torch.as_tensor(self.epsilon)
            self.lm_c = torch.as_tensor(self.lm_c)
        elif self.modeTrainC == trainC_exp:
            self.c = np.log(self.c)/ self.lm_c
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon = torch.as_tensor(self.epsilon)
            self.lm_c = torch.as_tensor(self.lm_c)

    def forward(self, x, y):
        
        if self.modeTrainC == trainC:
            c = torch.abs(self.c * self.lm_c) + self.epsilon 
        elif self.modeTrainC == trainC_exp:
            c = torch.exp(self.c * self.lm_c) + self.epsilon
        else:
            c = self.c
            
        x = torch.cat((x.view(x.size(0), -1), self.label_emb(y)), -1)

        layers = list(enumerate(self.cfg))

        for i, j in layers[:-1]:
            if j == 'h':
                x = self.layers[i](x, c)
                x = F.leaky_relu(x, 0.2)
            elif j == 'e':
                x = self.layers[i](x)
                x = F.leaky_relu(x, 0.2)
            else:
                x = self.layers[i](x, c)

        # end layer
        # end layer
        i, j = layers[-1]
        if j == 'e':
            x = self.layers[i](x)
        else:
            x = self.layers[i](x, c)

        return F.tanh(x)

    def get_real_c(self):
        
        if self.modeTrainC == noTrainC:
            return self.c
        elif self.modeTrainC == trainC:
            return self.c.item() * self.lm_c.item() + self.epsilon.item()
        elif self.modeTrainC == trainC_exp:
            return np.exp(self.c.item() * self.lm_c.item()) + self.epsilon.item()    
    
    
## Mix euclidean Hyperbolic GAN
class MixHypDiscriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    _defaults = {
        "c": 0.01,
        "modeTrainC": noTrainC,
        "nodeSizes": [2304, 4096, 2048, 1024, 1],
        "lm_c": 0.01, #0.00001,
        "epsilon": 1e-6,
        "activation": None
    }


    def __init__(self, **kwargs):
        super(MixHypDiscriminator, self).__init__()

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        nodeSizes = self.nodeSizes


        self.ln1 = hyply.MixHypEqualLinear(self.nodeSizes[0],
                                           self.nodeSizes[1],
                                           c=self.c,
                                           activation=self.activation)
        self.ln2 = hyply.MixHypEqualLinear(self.nodeSizes[1],
                                           nodeSizes[2],
                                           c=self.c,
                                           activation=self.activation)
        self.ln3 = hyply.MixHypEqualLinear(self.nodeSizes[2],
                                           self.nodeSizes[3],
                                           c=self.c,
                                           activation=self.activation)
        self.ln4 = hyply.MixHypEqualLinear(self.nodeSizes[3],
                                           self.nodeSizes[4],
                                           c=self.c,
                                           activation=None)
        
        if self.modeTrainC == trainC:
            self.c = self.c / self.lm_c
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon = torch.as_tensor(self.epsilon)
            self.lm_c = torch.as_tensor(self.lm_c)
            

    def forward(self, x):
        
        if self.modeTrainC == trainC:
            c = torch.abs(self.c * self.lm_c) + self.epsilon 
        else:
            c = self.c


        if self.activation==None or self.activation=='mobius_add':
            
            x = self.ln1(x, c=c)
            x = F.leaky_relu(x, 0.2)
            x = F.dropout(x, 0.1)
            
            x = self.ln2(x, c=c)
            x = F.leaky_relu(x, 0.2)
            x = F.dropout(x, 0.1)
            
            x = self.ln3(x, c=c)
            x = F.leaky_relu(x, 0.2)
            x = F.dropout(x, 0.1)
            
        else:
            x = self.ln1(x, c=c)
            x = F.dropout(x, 0.1)
            
            x = self.ln2(x, c=c)
            x = F.dropout(x, 0.1)
            
            x = self.ln3(x, c=c)
            x = F.dropout(x, 0.1)
            
        
        x = self.ln4(x, c=c)

        return x

    
## Generator
class MixHypGenerator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    _defaults = {
        "c": 0.01,
        "modeTrainC": noTrainC,
        "nodeSizes": [128, 1024, 2048, 4096, 2304],
        "lm_c": 0.01, #0.00001,
        "epsilon": 1e-6,
        "activation": None
    }

    def __init__(self, **kwargs):
        super(MixHypGenerator, self).__init__()

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.ln1 = hyply.MixHypEqualLinear(self.nodeSizes[0],
                                   self.nodeSizes[1],
                                   c=self.c,
                                   activation=self.activation)
        self.ln2 = hyply.MixHypEqualLinear(self.nodeSizes[1],
                                           self.nodeSizes[2],
                                           c=self.c,
                                           activation=self.activation)
        self.ln3 = hyply.MixHypEqualLinear(self.nodeSizes[2],
                                           self.nodeSizes[3],
                                           c=self.c,
                                           activation=self.activation)
        self.ln4 = hyply.MixHypEqualLinear(self.nodeSizes[3],
                                           self.nodeSizes[4],
                                           c=self.c,
                                           activation=None)
        
        if self.modeTrainC == trainC:
            self.c = self.c / self.lm_c
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon = torch.as_tensor(self.epsilon)
            self.lm_c = torch.as_tensor(self.lm_c)

    def forward(self, x):

        layers = list(enumerate(self.cfg))
        
        if self.modeTrainC == trainC:
            c = torch.abs(self.c * self.lm_c) + self.epsilon 
        else:
            c = self.c

        if self.activation==None or self.activation=='mobius_add':
            
            x = self.ln1(x, c=c)
            x = F.leaky_relu(x, 0.2)
            
            x = self.ln2(x, c=c)
            x = F.leaky_relu(x, 0.2)
            
            x = self.ln3(x, c=c)
            x = F.leaky_relu(x, 0.2)
            
        else:
            x = self.ln1(x, c=c)
            
            x = self.ln2(x, c=c)
            
            x = self.ln3(x, c=c)
            
        
        x = self.ln4(x, c=c)
        return F.tanh(x)