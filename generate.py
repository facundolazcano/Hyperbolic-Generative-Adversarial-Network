
import argparse
import os
import os.path as op

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import utils
from models.StyleGAN2 import Generator
from tqdm import tqdm


def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
           sample_z = torch.randn(args.sample, args.latent, device=device)

           sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent)

           utils.save_image(
            sample,
            op.join(args.path_sample, f'{str(i).zfill(6)}.png'),
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )


if __name__ == '__main__':
#    device = 'cuda:1'
#    torch.cuda.set_device(torch.device(device))

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="/home/azucar/Work/Tesis/Plots_and_diagrams/hyp_stylegan/500000pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--path_sample', type=str, default='/home/azucar/Work/Tesis/Plots_and_diagrams/hyp_stylegan/sample/')
    parser.add_argument('--cfg', type=str, default='eeeeeeee')
    parser.add_argument('--c', type=float, default=0.000001)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--activation_mix', type=int, default=0, help='activation of MixHypLineal, 0:fused_leaky_relu\n 1:hyp_bias_leaky_relu,\n 2:hpy_bias_leaky_relu_hyp_gain')
      
    args = parser.parse_args()

    os.mkdir(args.path_sample)
    print('path:', args.path_sample)
    print('seed: ', args.seed)
    print('device: ', args.device)

    device = 'cuda:'+str(args.device)
    torch.cuda.set_device(torch.device(device))
    cudnn.benchmark = False
    cudnn.deterministic=True
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    
    # set activation mix
    activation_mix_dict = {
        0: 'fused_leaky_relu',
        1: 'hyp_bias_leaky_relu',
        2: 'hyp_bias_leaky_relu_hyp_gain'
    }
    activation_mix = activation_mix_dict[args.activation_mix]

    args.latent = 512
    args.n_mlp = 8
    kwargs = {'n_mlp': 8,
              'style_dim': 512,
              'size': 32,
              'channel_multiplier': 2,
              'lr_mlp': 0.01,
              'cfg': args.cfg,
              'c': args.c,
              'activation_mix': activation_mix
   }

    g_ema = Generator(**kwargs).to(device)
    checkpoint = torch.load(args.ckpt, map_location=torch.device(device))
    print(g_ema.style)

    print(checkpoint.keys())
    #print(checkpoint['g'].keys())
    g_ema.load_state_dict(checkpoint['g_ema'])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
