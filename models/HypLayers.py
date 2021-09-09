


import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

import models.hyptorch.pmath as pmath


class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        mv = pmath.mobius_matvec(self.weight, x, c=c)
        if self.bias is None:
            return pmath.project(mv, c=c)
        else:
            bias = pmath.expmap0(self.bias, c=c)
            return pmath.project(pmath.mobius_add(mv, bias), c=c)


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )


class HypEqualizedLinear(nn.Module):
    """Hyp Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, c,  gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True):
        super(HypEqualizedLinear, self).__init__()

        self.in_features = input_size
        self.out_features = output_size
        self.c = c
        self.bias_flag = bias

        #self.b_mul = torch.as_tensor(lrmul)

        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul * 0.1

        self.w_mul = torch.as_tensor(self.w_mul)
        #print(self.w_mul)
        self.weight = torch.nn.Parameter(data=torch.randn(output_size, input_size) * init_std)

        if bias:
            self.bias = nn.Parameter(data=torch.zeros(self.out_features))
            self.b_mul = torch.as_tensor(lrmul* 0.1)
        else:
             self.register_parameter('bias', None)


    def forward(self, x, c=None):

        if c is None:
            c = self.c

        w_lr = self.weight *  self.w_mul
        mv = pmath.mobius_matvec(w_lr, x, c=c)

        if self.bias is None:
            out = pmath.project(mv, c=c)
        else:
            bias = self.bias * self.b_mul
            bias = pmath.expmap0(bias, c=c)
            out = pmath.project(pmath.mobius_add(mv, bias), c=c)
        out = F.leaky_relu(out, 0.2)
        return out
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    """
    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError("if train_x=True, ball_dim has to be integer, got {}".format(ball_dim))
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter('xp', None)
                   
        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))            
        else:
            self.c = c

        self.train_x = train_x
        self.train_c = train_c

    def forward(self, x, c=None):
        
        if self.train_c or c is None:
            c = self.c
        
        
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=c), c=c)
            return pmath.project(pmath.expmap(xp, x, c=c), c=c)
        return pmath.project(pmath.expmap0(x, c=c), c=c)

    def extra_repr(self):
        return 'c={}, train_x={}'.format(self.c, self.train_x)


class FromPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Poincare ball
    to n-dim Euclidean space
    """
    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):

        super(FromPoincare, self).__init__()

        if train_x:
            if ball_dim is None:
                raise ValueError("if train_x=True, ball_dim has to be integer, got {}".format(ball_dim))
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter('xp', None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_c = train_c
        self.train_x = train_x

    def forward(self, x, c=None):
        if self.train_c or c is None:
            c = self.c
            
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=c), c=c)
            return pmath.logmap(xp, x, c=c)
        return pmath.logmap0(x, c=c)

    def extra_repr(self):
        return 'train_c={}, train_x={}'.format(self.train_c, self.train_x)


class HypEqualLinear(nn.Module):
    """Hyp Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, c,
                 gain=2 ** 0.5, lr_mul=1, bias=True):
        super(HypEqualLinear, self).__init__()

        self.in_features = input_size
        self.out_features = output_size
        self.c = c
        self.bias_flag = bias
        
        # set parameters
        self.weight = nn.Parameter(torch.randn(output_size, input_size).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter('bias', None)

        # scale factors
        self.scale = (1 / math.sqrt(input_size)) * lr_mul
        self.scale = torch.as_tensor(self.scale)
        self.lr_mul = torch.as_tensor(lr_mul)
        self.gain = torch.as_tensor(gain)

    def forward(self, x, c=None):

        if c is None:
            c = self.c

        #w_lr = self.weight *  self.w_mul
        mv = pmath.mobius_matvec(self.weight * self.scale, x, c=c)

        if self.bias is None:
            out = pmath.project(mv, c=c)
        else:
            bias = self.bias * self.lr_mul
            bias = pmath.expmap0(bias, c=c)
            out = pmath.project(pmath.mobius_add(mv, bias), c=c)
        out = F.leaky_relu(out, 0.2)
        out = pmath.mobius_scalar_mult(out, self.gain, c=c)
        return out
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )


    
## own version of Hyp linear called Mixed Hyperbolic Equilized Linear
class MixHypEqualLinear(nn.Module):
    """MIX Hyp Linear layer with equalized learning rate and custom learning rate multiplier."""
    

    def __init__(self, input_size, output_size, c,
                 lr_mul=1, lrmul_c=1, gain=2 ** 0.5,
                 epsilon_c=1e-6, activation=None,
                 bias=True, train_c=False):
        
        super(MixHypEqualLinear, self).__init__()

        self.in_features = input_size
        self.out_features = output_size
        self.bias_flag = bias

        self.weight = nn.Parameter(torch.randn(output_size, input_size).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(data=torch.zeros(self.out_features))
        else:
            self.register_parameter('bias', None)
        
        
        self.activation = activation
        
        # scale factors
        self.scale = (1 / math.sqrt(input_size)) * lr_mul
        self.scale = torch.as_tensor(self.scale)
        self.lr_mul = torch.as_tensor(lr_mul)
        self.gain = torch.as_tensor(gain)
        
        
        # train C option
        self.train_c = train_c
        if train_c:
            self.c = c / lrmul_c
            self.c = nn.Parameter(torch.Tensor([self.c,]))
            self.epsilon_c = torch.as_tensor(epsilon_c)
            self.lrmul_c = torch.as_tensor(lrmul_c)
        else:
            self.c = c
               

    def forward(self, x, c=None):

        if self.train_c:
            c = torch.abs(self.c * self.lrmul_c) + self.epsilon_c
        elif c is None:
            c = self.c
                
        if not(self.bias_flag):
            out = F.linear(x, self.weight * self.scale)
            out = pmath.project(pmath.expmap0(out, c=c), c=c)
                    
        elif self.activation=='fused_leaky_relu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
            out = pmath.project(pmath.expmap0(out, c=c), c=c)
           
        elif self.activation=='hyp_bias':
            out = F.linear(x, self.weight * self.scale)
            out = pmath.project(pmath.expmap0(out, c=c), c=c)
            
            bias = self.bias * self.lr_mul
            bias = pmath.project(pmath.expmap0(bias, c=c), c=c)
            
            out = pmath.project(pmath.mobius_add(out, bias), c=c)
            
        elif self.activation=='hyp_bias_leaky_relu':
            out = F.linear(x, self.weight * self.scale)
            out = pmath.project(pmath.expmap0(out, c=c), c=c)
            
            bias = self.bias * self.lr_mul
            bias = pmath.project(pmath.expmap0(bias, c=c), c=c)
            
            out = pmath.project(pmath.mobius_add(out, bias), c=c)
            out = F.leaky_relu(out, 0.2) * self.gain
            
        elif self.activation=='hyp_bias_leaky_relu_hyp_gain':
            out = F.linear(x, self.weight * self.scale)
            out = pmath.project(pmath.expmap0(out, c=c), c=c)
            
            bias = self.bias * self.lr_mul
            bias = pmath.project(pmath.expmap0(bias, c=c), c=c)
            
            out = pmath.project(pmath.mobius_add(out, bias), c=c)
            out = F.leaky_relu(out, 0.2)
            out = pmath.mobius_scalar_mult(out, self.gain, c=c)
            
            
            
        elif self.activation is None:
            out = F.linear(
                 x, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
            out = pmath.project(pmath.expmap0(out, c=c), c=c)
            
        else:
            raise Exception("activation is not defined")
            
        return out
            
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}, activation={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.c, self.activation
        )
