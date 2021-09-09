"""
-------------------------------------------------
   File Name:    GAN.py
   Author:       Zhonghao Huang
   Date:         2019/10/17
   Description:  Modified from:
                 https://github.com/akanimax/pro_gan_pytorch
                 https://github.com/lernapparat/lernapparat
                 https://github.com/NVlabs/stylegan
-------------------------------------------------
"""

import os
import datetime
import time
import timeit
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn

#import models.hyptorch.nn as hypnn

from models.Layers import PixelNorm, EqualLinear, ConstantInput, StyledConv, ConvLayer, ResBlock, ToRGB
from models import HypLayers as hyply #HypEqualLinear, MixHypEqualLinear
from .hgan import noTrainC, trainC, trainC_exp
#print('notrainc', noTrainC)
print(1e-6)

class StyleMapping(nn.Module):
    
    def __init__(self, style_dim, n_mlp, lr_mlp,
                 cfg='eeeeeeee', c=1, activation_mix='fused_leaky_relu',
                 mode_train_c=noTrainC,  epsilon=1e-6, lm_c=0.01):
        super(StyleMapping, self).__init__()
        
        self.modeTrainC = mode_train_c
        self.c = c
        self.cfg = cfg
        self.activation_mix = activation_mix
        self.lr_mlp = lr_mlp
        self.lm_c = lm_c
        self.epsilon = epsilon
        
        
        
        self.layers = nn.ModuleList()
        
        self.layers.append(PixelNorm())

        for i in cfg:

            if i == 'e':
                self.layers.append(
                    EqualLinear(
                        style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                    )
                )
            elif i == 'h':
                self.layers.append(
                    hyply.HypEqualLinear(
                        style_dim, style_dim, lr_mul=lr_mlp, c=c)
                )
            elif i == 'm':
                self.layers.append(
                    hyply.MixHypEqualLinear(
                        style_dim, style_dim, lr_mul=lr_mlp, c=c, 
                        activation=activation_mix
                    )
                )
            elif i == 't':
                self.layers.append(hyply.ToPoincare(c=c,
                                               train_x=False,
                                               train_c=False))
            elif i == 'f':
                self.layers.append(hyply.FromPoincare(c=c,
                                                 train_x=False,
                                                 train_c=False))
            elif i == 'p':
                self.layers.append(PixelNorm())

            else:
                raise ValueError('not implemented option layer')
            
            
            
        # mode train C
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
            

    def forward(self, noise_input):
        
        # set c value
        if self.modeTrainC == trainC:
            c = torch.abs(self.c * self.lm_c) + self.epsilon
        elif self.modeTrainC == trainC_exp:
            c = torch.exp(self.c * self.lm_c) + self.epsilon
        else:
            c = self.c
            
        # evualuate model
        
        # First layer -> pixel norm
        x = self.layers[0](noise_input)
        
        layers = list(enumerate(self.cfg))
        for i, j in layers:
            # if layers has a c value as input
            if j == 'h' or j == 'm' or j=='f' or j=='t':
                x = self.layers[i+1](x, c)                        
            else:
                x = self.layers[i+1](x)
        return x

    
    def get_real_c(self):
        
        if self.modeTrainC == noTrainC:
            return self.c
        elif self.modeTrainC == trainC:
            return self.c.item() * self.lm_c.item() + self.epsilon.item()
        elif self.modeTrainC == trainC_exp:
            return np.exp(self.c.item() * self.lm_c.item()) + self.epsilon.item()



class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        cfg='eeeeeeee',
        c=0.0001,
        activation_mix='fused_leaky_relu',
        mode_train_c = noTrainC,
        lrml_c=0.01
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim



        self.style = StyleMapping(style_dim,
                                  n_mlp,
                                  lr_mlp,
                                  cfg=cfg,
                                  c=c,
                                  activation_mix=activation_mix,
                                  mode_train_c=mode_train_c,
                                  epsilon=1e-6,
                                  lm_c=lrml_c)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
             res = (layer_idx + 5) // 2
             shape = [1, 1, 2 ** res, 2 ** res]
             self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


if __name__ == '__main__':
    print('Done.')
