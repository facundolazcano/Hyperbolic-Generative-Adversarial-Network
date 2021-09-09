import os
import numpy as np
import errno
import torchvision.utils as vutils
#from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
from torch.autograd.variable import Variable
import pynvml


'''
    TensorBoard Data will be stored in './runs' path
'''

'''
class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/D_error'.format(self.comment), d_error, step)
        self.writer.add_scalar(
            '{}/G_error'.format(self.comment), g_error, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        
        #input images are expected in format (NCHW)

        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        
        if format=='NHWC':
            images = images.transpose(1,3)
        

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)

        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
        
        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data
        
        
        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch,num_epochs, n_batch, num_batches)
             )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

'''

def images_to_vectors(images, sizeImageVect):
    return images.view(images.size(0), sizeImageVect)


def vectors_to_images(vectors, dimImage):
    return vectors.view(vectors.size(0), dimImage[0], dimImage[1], dimImage[2])


def noise(size, dim=100):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, dim, dtype=torch.double))
    return n


def random_labels(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    fake_labels = Variable(torch.randint(0, 10, (size,), dtype=torch.long))
    return fake_labels


def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1, dtype=torch.double))
    return data


def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1, dtype=torch.double))
    return data


# write config json
def write_config_json():
    import json


    data= {}
    data['c_list']=  [1e-3 ,1e-4, 1e-5, 1e-6]

    data['arch_dis'] = [
        'ethfee',
        'ethhfe',
        'eethfe'
        #'thhhh'

        #    'eethfe',
        #    'ethfee',
        #    'ethhfe',
        #    'thhhhf',
        #    'thhhfe',
        #    'thhfee',
        #    'thfeee'
    ]

    data['arch_gen'] = [
        'eeee'
        #'ethhfe',
        #'eethfe'
        #'ethfee',
        #'ethhfe'
    ]

    with open('configs/23-07-20_new_experiments_3.json', 'w') as outfile:
        json.dump(data, outfile)


## Read Experiments
def read_experiment(path_experiment):

    # previus imports
    import pandas as pd
    import os.path as op

    result = dict()
    for root, dirs, files in os.walk(path_experiment, topdown=True):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == '.csv':
                data = pd.read_csv(op.join(root, file))
                result[root.replace(path_experiment, '')] = data

    print('Done!')
    return result


def get_data_events(summary, tag, len_data=None):

    
    data = []
    
    if len_data is None:
        len_data = float('inf')
    
    # iterate summary
    while len(data)< len_data:
        
        try:
            event = next(summary)
            #print(event)
        except StopIteration:
            break
        except:
            print(event)
            continue
            
        for v in event.summary.value:
            #print(v)
            if v.tag == tag:
                data.append(v.simple_value)
    return data



def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed



def gradient_2_tb(model, writer, name_var, iter_train):

#gradients = torch_grad(outputs=net_output, inputs=net_input,
#		grad_outputs=torch.ones(net_output.size()).to(device=device).double(),
#		create_graph = True, retain_graph = True, only_inputs=True)[0]

#    gradients_norm = torch.sqrt(torch.sum(gradients**2).item() + 1e-12)
    params = model.state_dict()
    for name, param in zip(params.keys(), model.parameters()):
        gradient = param.grad
        gradient_norm = gradient.norm().item() + 1e-12
        writer.add_scalar('_'.join([name_var, 'layer', name]), gradient_norm, iter_train)


def gpu_used_memory(div_id=0):
    gpu_used_memory = 0
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(div_id)
    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        if proc.pid == os.getpid():
            gpu_used_memory = proc.usedGpuMemory
