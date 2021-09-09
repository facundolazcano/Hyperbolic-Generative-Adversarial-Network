import argparse
import math
import random
import os
import os.path as op

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.distributed as dist
from torchvision import utils as vutils

from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import geoopt

from models.StyleGAN2_train_c import Generator, Discriminator
from models.Losses import d_r1_loss, g_nonsaturating_loss, g_path_regularize, d_logistic_loss
from utils.utils_stylegan import accumulate, requires_grad, mixing_noise
from utils.utils_dataset import loader_CIFAR10, sample_data


from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from models.non_leaking import augment


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, writer):

    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # set init state for values
    mean_path_length = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    # distributed training?
    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    #ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = 0.0#args.augment_p if args.augment_p > 0 else 0.0
    #ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    # Training Loop
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img[0].to(device)
        real_img.requires_grad = True
        
        ### TRAIN DISCRIMINATOR ###
        
        # clean grads
        discriminator.zero_grad()
        generator.zero_grad()
        
        # set requiere grads
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch_size, args.latent, args.mixing, device)
        d_loss, real_pred, fake_pred = d_logistic_loss(generator, discriminator, noise, real_img)

        # apply r1 regularization discriminator
        if i % args.d_reg_every == 0:
            r1_loss = d_r1_loss(real_pred, real_img, gamma=args.r1)
            d_loss += r1_loss * args.d_reg_every
            
        # apply optimization step
        d_loss.backward()
        d_optim.step()
        
        # record results
        loss_dict["r1"] = r1_loss
        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()
        
        ### TRAIN GENERATOR ###
        
        # clean grads
        discriminator.zero_grad()
        generator.zero_grad()

        # set requiere grads
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch_size, args.latent, args.mixing, device)
            
        g_loss = g_nonsaturating_loss(discriminator, generator, noise)

        # apply path regularization generator
        if (i % args.g_reg_every == 0) and not(args.not_g_reg):
            path_batch_size = max(1, args.batch_size // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            path_loss, mean_path_length, path_lengths = g_path_regularize(generator,
                                                                          noise,
                                                                          mean_path_length,
                                                                          path_batch_size,
                                                                          args.path_regularize
                                                                         )

            g_loss += path_loss * args.g_reg_every
            
            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )
        
        # apply optimization step
        g_loss.backward()
        g_optim.step()

        # record results
        loss_dict["g"] = g_loss
        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        # ema calculation
        accumulate(g_ema, g_module, accum)

        # pass results
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )
            # write in tensorboard
            writer.add_scalar('Loss/D', d_loss, i)
            writer.add_scalar('Loss/D_r1', r1_val, i)
            writer.add_scalar('Loss/G', g_loss, i)
            writer.add_scalar('Loss/G_pl', path_loss_val, i)
            writer.add_scalar('D_x', real_score_val, i)
            writer.add_scalar('D_G_z', fake_score_val, i)
            writer.add_scalar('path_length', path_length_val, i)
            if args.mode_train_c==1 or args.mode_train_c==2:
                writer.add_scalar('C_g', generator.style.get_real_c(), i)
            
            # debug error
            #cuda_memory = torch.cuda.memory_allocated(device)
            #writer.add_scalar('memory_usage', cuda_memory, i)


            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    vutils.save_image(
                        sample,
                        op.join(args.path, f"sample/{str(i).zfill(6)}.png"),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    op.join(args.path, f"checkpoint/{str(i).zfill(6)}.pt"),
                )


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default='/home/jenny2/HypStyleGan')
    parser.add_argument("--iter", type=int, default=200001)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--r1", type=float, default=0.1)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--cfg", type=str, default='thfeeeeeee')
    parser.add_argument("--c", type=float, default='0.01')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--bias_false", action="store_true", help='bias of the style mapping')
    parser.add_argument("--radam", action="store_true", help='train with radam?')
    parser.add_argument("--not_g_reg", action="store_true", help='regularization generator?')
    parser.add_argument('--activation_mix', type=int, default=0, help='activation of MixHypLineal, 0:fused_leaky_relu\n 1:hyp_bias_leaky_relu, 2:hpy_bias_leaky_relu_hyp_gain')
    parser.add_argument('--mode_train_c', type=int, default=0, help='0: do not train(defualt)\n 1:train c with abs(c*lr)+e\n 2: train c with exp(c*lr)+e in generator.')
    parser.add_argument("--lrml_c", type=float, default=0.01)


    args = parser.parse_args()
    
    print(args.mixing)
    print(args.seed)
    # create dir tensorboard
    path_tb = op.join(args.path, 'tensorboard')
    writer = SummaryWriter(path_tb)
    writer.add_text('args', str(args), 0)
    writer.add_text('Random Seed: ', str(args.seed), 0)

    #n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = 0
    device = 'cuda:' + str(args.device)
    torch.cuda.set_device(torch.device(device))
    cudnn.benchmark = False
    cudnn.deterministic=True
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print('device:', args.device)
    print('augment:', args.augment)

    # make dirs for save models and sampled
    os.makedirs(args.path, exist_ok=True)
    os.makedirs(op.join(args.path, 'sample'), exist_ok=True)
    os.makedirs(op.join(args.path, 'checkpoint'), exist_ok=True)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        
    # set activation mix
    activation_mix_dict = {
        0: 'fused_leaky_relu',
        1: 'hyp_bias_leaky_relu',
        2: 'hyp_bias_leaky_relu_hyp_gain'
    }
    activation_mix = activation_mix_dict[args.activation_mix]
    
    # overload args
    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0
    
    print(args.cfg)
    
    generator = Generator(
        args.size, args.latent,
        args.n_mlp, channel_multiplier=args.channel_multiplier,
        cfg=args.cfg, c=args.c,
        activation_mix=activation_mix,
        mode_train_c=args.mode_train_c, lrml_c=args.lrml_c
    ).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)

    print('generator style')
    print(generator.style)
    print('lrml_c:', generator.style.lm_c)
    #print('lrml_c:', generator.lrml_c)
    #print('discriminator')
    #print(discriminator)

    g_ema = Generator(
        args.size, args.latent,
        args.n_mlp, channel_multiplier=args.channel_multiplier,
        cfg=args.cfg, c=args.c,
        activation_mix=activation_mix,
        mode_train_c=args.mode_train_c, lrml_c=args.lrml_c
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    #print(torch.cuda.memory_summary())
    #print(torch.cuda.memory_allocated())
    
    if args.radam:
        g_optim = geoopt.optim.RiemannianAdam(
            generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )                         
    else:
        g_optim = optim.Adam(
            generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = op.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    loader = loader_CIFAR10(batch_size=args.batch_size)

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, writer)
