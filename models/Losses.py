#from torch import nn, autograd, optim
import math
import torch
from torch.nn import functional as F
from torch import autograd

#from HGAN.utils.utils_stylegan import mixing_noise


def d_logistic_loss(generator, discriminator, noise, real_img, augment=None):
        
    # gen images
    fake_img, _ = generator(noise)
         
    if augment:
        print('not implemented')

    fake_score= discriminator(fake_img)
    real_score = discriminator(real_img)
      
    real_loss = F.softplus(-real_score)
    fake_loss = F.softplus(fake_score)

    return real_loss.mean() + fake_loss.mean(), real_loss, fake_loss



def d_r1_loss(real_score, real_img, gamma=10):
    grad_real, = autograd.grad(
        outputs=real_score.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    reg = grad_penalty * (gamma / 2)

    return reg


def g_nonsaturating_loss(discriminator, generator, noise):
    fake_img, _ = generator(noise)
    fake_score = discriminator(fake_img)
    loss = F.softplus(-fake_score).mean()

    return loss


def g_path_regularize(generator, noise, mean_path_length, path_batch_size, pl_weight=2.0, decay=0.01):
    
    fake_img, latents = generator(noise, return_latents=True)
    
    
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2)
    path_penalty = (path_penalty * pl_weight).mean()

    return path_penalty, path_mean.detach(), path_lengths


def g_wgan(generator, discriminator, noise):
    
    fake_img = generator(noise)
    fake_score = discriminator(fake_img)
    loss = - prediction.mean()
    return loss


def d_wgan(generator, discriminator, noise, real_img, wgan_epsilon=0.001):
    
    fake_img = generator(noise)
    real_score = discriminator(real_img)
    fake_score = discriminator(fake_img)
    loss = fake_score.mean() - real_score.mean()
    # epsilon regulrization not implemented
    return loss
    
    
    
def d_wgan_gp(generator, discriminator, noise, real_img, device, wgan_epsilon=0.001, gp_weight=10.0):
    
    fake_img = generator(noise)
    real_score = discriminator(real_img)
    fake_score = discriminator(fake_img)
    loss = fake_score.mean() - real_score.mean()
    
    # epsilon regulrization not implemented
    
    # Gradient Penalty
    N = real_img.size(0)
    alpha = torch.rand(N, 1).to(device=device)
    interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
    interpolated.required_grad()
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device=device),
                           create_graph = True, retain_graph = True, only_inputs=True)[0]
    
    gradients = gradients.view(N, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = gp_weight * ((gradients_norm - 1) ** 2).mean()
    loss += gradient_penalty
    return loss


#def d_logistic_loss(generator, discriminator, noise, real_img)
    
#    fake_img = generator(noise)
#    real_score = discriminator(real_img)
#    fake_score = discriminator(fake_img)
    
    
    
    
    
    
    