import os
import sys
import numpy as np
import torch
import torch.nn.functional as F


from models.StyleGAN2 import Generator, Discriminator



iterations = 10


ckpt_path = '/home/jenny2/HypStyleGAN/Nuevos_exp/eeee_eeee_tuning/checkpoint/200000.pt'

ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
device=3


generator_1 = Generator(
    32, 512,
    8, channel_multiplier=2,
    cfg='eeeeeeeet', c=0.0001
)

generator_2 = Generator(
    32, 512,
    8, channel_multiplier=2,
    cfg='eeeeeeem', c=0.0001
)




generator_1.load_state_dict(ckpt['g_ema'])
generator_2.load_state_dict(ckpt['g_ema'])




with torch.no_grad():

    for i in range(iterations):
        print('iteartion:', i)
        sample_z = torch.randn(1, 512)


        w_1 = generator_1.style(sample_z)
        w_2 = generator_2.style(sample_z)
        w_1_2 = generator_1.style(sample_z)
        w_2_2 = generator_2.style(sample_z)


        print('w_1, norm:',w_1.norm(), ' mean:', w_1.mean())
        print('w_2, norm:',w_2.norm(), ' mean:', w_2.mean())
        print('w_1_2, norm:',w_1.norm(), ' mean:', w_1_2.mean())
        print('w_2_2, norm:',w_2_2.norm(), ' mean:', w_2_2.mean())
