import argparse
import os
import logging
import random
import numpy as np
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torch.autograd import grad as torch_grad
from torchvision import transforms, datasets
from torchvision.utils import save_image
from scipy.io import savemat
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.autograd.variable import Variable

#from my_dataset import TorontoFaceDataset
from utils.utils import images_to_vectors, vectors_to_images, \
    noise, ones_target, zeros_target, random_labels, gradient_2_tb, gpu_used_memory
from utils import utils_dataset
from models import hgan
import geoopt
#from torchsummary import summary

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch HGAN')
    parser.add_argument('--root', type=str, default='/home/jenny2/HCGAN/D_eeee_G_eeee',
                        help='root of result file')

    parser.add_argument('--pre_train', nargs='+', default=None,
                        help='path discriminator and path generator')

    parser.add_argument('--name', type=str, default='D_eeee_G_eeee_lr_1e-3',
                        help='name of model, file for save result for model')
    parser.add_argument('--device', type=int, default=2,
                        help='index of devices where hgan is training')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--epochs_sample', type=int, default=10, metavar='N',
                        help='number of epochs to sample (default: 10)')
    parser.add_argument('--epochs_save', type=int, default=10,
                        help='number of epochs to saves models (default: 50)')
    # optimizator params
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--b1', type=float, default=0.5, metavar='B1',
                        help='beta 1 (default: 0.51)')
    parser.add_argument('--b2', type=float, default=0.999, metavar='B2',
                        help='beta 2 (default: 0.999)')
    parser.add_argument('--gp_weight', type=float, default=10,
                        help='gp weight (default: 10)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay of Adam Optimizer')

    parser.add_argument('--train_c', type=bool, default=False, help='Flag to train c -> 1/r')
    
    parser.add_argument('--bias', type=bool, default=False, help='bias is False or True?')
    
    parser.add_argument('--opt_D', type=str, default='adam', help='adam or radam')
    
    parser.add_argument('--opt_G', type=str, default='adam', help='adam or radam')

    parser.add_argument('--cfg_d', type=str, default='eeee',
                        help='layes in the net')

    parser.add_argument('--cfg_g', type=str, default='eeee',
                        help='layes in the net')

    parser.add_argument('--c_d', type=float, default=1e-3, metavar='C_d',
                        help='C of hiperbolic space of discriminator (default: 0.001)')
    
    parser.add_argument('--c_g', type=float, default=1e-5, metavar='C_g',
                        help='C of hiperbolic space of generator (default: 0.001)')

    parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, CIFAR10, TFD48')
    parser.add_argument('--node_sizes_d', nargs='+', type=int, default=None)
    parser.add_argument('--node_sizes_g', nargs='+', type=int, default=None)
    
    parser.add_argument('--mode_train_c_g', type=int, default=0, help='0: do not train(defualt)\n 1:train c with abs(c*lr)+e\n 2: train c with exp(c*lr)+e in generator.')
    parser.add_argument('--mode_train_c_d', type=int, default=0, help='0: do not train(defualt)\n 1:train c with abs(c*lr)+e\n 2: train c with exp(c*lr)+e in discriminator.')
    parser.add_argument('--lrmul_c', type=float, default=0.01, metavar='L',
                        help='laerning rate multiplier for train c(defualt 0.01)')

    args = parser.parse_args()

    # files to save
    # make results/ directory, if necessary
    if not os.path.exists(args.root):
        os.makedirs(args.root)
    # make files
    path_images = os.path.join(args.root, args.name, 'images')
    path_tb = os.path.join(args.root, args.name, 'tensorboard')
    path_mats = os.path.join(args.root, args.name, 'mats')
    path_model = os.path.join(args.root, args.name, 'models')

    os.makedirs(path_images)
    os.makedirs(path_mats)
    os.makedirs(path_model)

    # config logger
    logging.basicConfig(filename=os.path.join(args.root, args.name, 'info.log'),
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    #log argparse
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    # create dir tensorboard
    writer = SummaryWriter(path_tb)

    # opt = parser.parse_args()
    # print(opt)
    writer.add_text('args', str(args), 0)
    writer.add_text('Random Seed: ', str(args.seed), 0)
    cudnn.benchmark = True
    
    torch.cuda.set_device(torch.device("cuda:"+str(args.device)))
    
    # flag of use cuda device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    logging.info('use_cuda:' + str(use_cuda))

    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # set device
    device = torch.device("cuda:"+str(args.device) if use_cuda else "cpu")

    # load dataset
    if args.dataset == 'TFD48':
        dataloader = utils_dataset.loader_TFD48(args.batch_size)
        # nodes of network
        node_sizes_d = hgan.nodeSizesDiscriminatorTFD48
        node_sizes_g = hgan.nodeSizesGeneratorTFD48
        dimImage = (1, 48, 48)
        sizeImageVect = 2304

    elif args.dataset == 'CIFAR10':
        dataloader = utils_dataset.loader_CIFAR10(args.batch_size)
        # nodes of network
        node_sizes_d = hgan.nodeSizesDiscriminatorCIFAR10
        node_sizes_g = hgan.nodeSizesGeneratorCIFAR10
        dimImage = (3, 32, 32)
        sizeImageVect = 3072

    elif args.dataset == 'MNIST':
        dataloader = utils_dataset.loader_MNIST(args.batch_size)
        # nodes of network
        node_sizes_d = hgan.nodeSizesDiscriminatorMNIST
        node_sizes_g = hgan.nodeSizesGeneratorMNIST
        dimImage = (1, 28,28)
        sizeImageVect = 784

    else:
        raise ValueError('dataset:' + args.dataset + 'not implemented. look help')

    logging.info('Loaded ' + args.dataset)

    # num of iterations per epochs
    num_batches = len(dataloader)

    # load nets
    if args.train_c:
        print('not implmented option')
        #kwargs_d = {'modeTrainC': trainC}

    # Make Discriminator network
    # load param c ball Poincaré
    kwargs = { 'c' : args.c_d }
    kwargs['bias'] = args.bias
    kwargs['cfg'] = args.cfg_d
    kwargs['modeTrainC'] = args.mode_train_c_d
    kwargs['lm_c'] = args.lrmul_c

    # node sizes
    if args.node_sizes_d == None:
        kwargs['nodeSizes'] = node_sizes_d
    else:
        kwargs['nodeSizes'] = args.node_sizes_d

    # create Discriminator
    discriminator = hgan.HypConditionalDiscriminator(**kwargs)

    #logg info
    logging.info('Load Discriminator network with cfg' + args.cfg_d)
    logging.info(discriminator)
    print(discriminator)

    # Make Generator network
    kwargs = { 'c' : args.c_g } 
    kwargs['bias'] = args.bias
    kwargs['cfg'] = args.cfg_g
    kwargs['modeTrainC'] = args.mode_train_c_g
    kwargs['lm_c'] = args.lrmul_c

    # nodes sizes
    if args.node_sizes_g == None:
        kwargs['nodeSizes'] = node_sizes_g
    else:
        kwargs['nodeSizes'] = args.node_sizes_g

    # create generator network
    generator = hgan.HypConditionalGenerator(**kwargs)

    #log info
    logging.info('Load Generator  network with cfg' + args.cfg_g)
    logging.info(generator)
    print(generator)

    # load trained models
    if args.pre_train:
        logging.info('load pretrained models')
        discriminator.load_state_dict(torch.load(args.pre_train[0]))
        generator.load_state_dict(torch.load(args.pre_train[1]))

    # pass to double
    discriminator.double()
    generator.double()

    # to device
    discriminator.to(device=device)
    generator.to(device=device)

    # optimizers
    if args.opt_D=='adam':
        d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2),
                                 weight_decay=args.weight_decay)
    else:
        d_optimizer = geoopt.optim.RiemannianAdam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2),
                                 weight_decay=args.weight_decay)
        print('It is using radam on discriminator')
        
    if args.opt_G=='adam':
        g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2),
                                 weight_decay=args.weight_decay)
    else:
        g_optimizer = geoopt.optim.RiemannianAdam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2),
                                 weight_decay=args.weight_decay)
        print('It is using radam on generator')

    # loss
    loss = nn.BCEWithLogitsLoss()

    # fixed noise for test images
    test_noise = noise(10000, 128).to(device=device)
    show_labels = Variable(torch.from_numpy(np.arange(10)).to(dtype=torch.long).to(device=device))
    print(test_noise[0:10].shape, show_labels)
    test_labels = random_labels(10000).to(device=device)
    print(test_labels.shape)
    losses = []


    # train
    logging.info('Start to Train')
    print('Start to Train')
    iter_train = 0
    for epoch in range(args.epochs):
        for n_batch, real_batch in enumerate(dataloader):

            #for name, param in discriminator.named_parameters():
            #    if param.requires_grad:
            #        print(name, param.data)}
            labels = real_batch[1]
            real_batch = real_batch[0]

            N = real_batch.size(0)
            real_batch = real_batch.double()

            # 1. Train Discriminator
            labels = Variable(labels.long()).to(device=device)
            real_data = Variable(images_to_vectors(real_batch, sizeImageVect)).to(device=device)

            # Generate fake data and detach
            # (so gradients are not calculated for generator)
            fake_labels = random_labels(N).to(device=device)
            fake_data = generator(noise(N, 128).to(device=device), fake_labels).detach()

            # Train D

            # Reset gradients
            d_optimizer.zero_grad()

            # 1.1 Train on Real Data
            prediction_real = discriminator(real_data, labels)
                       
            # Calculate error and backpropagate
            error_real = loss(prediction_real, ones_target(N).to(device=device))
            error_real.backward()

            # 1.2 Train on Fake Data
            prediction_fake = discriminator(fake_data, fake_labels)
            
            # Calculate error and backpropagate
            error_fake = loss(prediction_fake, zeros_target(N).to(device=device))
            error_fake.backward()

            gradient_2_tb(discriminator, writer=writer, iter_train=iter_train, name_var='D_grad')

            # 1.3 Update weights with gradients
            d_optimizer.step()
            d_loss = error_fake + error_real

            # 2. Train Generator

            # Generate fake data
            fake_labels = random_labels(N).to(device=device)
            ruido = noise(N, 128).to(device=device)
 
            # Train G
            # Reset gradients
            g_optimizer.zero_grad()
            fake_data = generator(ruido, fake_labels)


            # Sample noise and generate fake data
            prediction = discriminator(fake_data, fake_labels)

            # Calculate error and backpropagate
            g_loss = loss(prediction, ones_target(N).to(device=device))
            g_loss.backward()

            #data del gradiente al tensorboardX
            gradient_2_tb(generator, writer=writer, iter_train=iter_train, name_var='G_grad')

            # Update weights with gradients
            g_optimizer.step()

            # write in tensorboard
            writer.add_scalar('Loss/D', d_loss.item(), iter_train)
            writer.add_scalar('Loss/G', g_loss.item(), iter_train)

            cuda_memory = gpu_used_memory(args.device)
            writer.add_scalar('memory_usage', cuda_memory, iter_train)

            if args.mode_train_c_d != 0:
                writer.add_scalar('C_d', discriminator.get_real_c(), iter_train)
            if args.mode_train_c_g != 0:
                writer.add_scalar('C_g', generator.get_real_c(), iter_train)
            iter_train +=1

        # logg avance
        logging.info(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.epochs, d_loss.item(), g_loss.item())
        )
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.epochs, d_loss.item(), g_loss.item())
        )

        # Display Progress every few batches and save
        if (epoch + 1) % args.epochs_sample == 0:
            # save mat
            logging.info('save image')
            vect = generator(test_noise, test_labels).detach().to(device='cpu')
            savemat(path_mats + "/" + str(epoch + 1) + '.mat', {'images': vect.numpy()})

            # np.np.savez_compressed('/tmp/123', a=test_array, b=test_vector)
            # save test_images
            #show_labels = Variable(torch.from_numpy(np.arange(9)))
            vect = generator(test_noise[0: 10], show_labels).detach().to(device='cpu')
            test_images = vectors_to_images(vect[0:16], dimImage)
            save_image(test_images.data, path_images + "/%d.png" % (epoch + 1), nrow=4, normalize=True)

            ## save model
        if args.save_model and (epoch + 1) % args.epochs_save == 0:
            logging.info('save model')
            torch.save(discriminator.state_dict(), path_model + "/discriminator" + ".pt")
            torch.save(generator.state_dict(), path_model + "/generator" +  ".pt")
    torch.cuda.empty_cache()

