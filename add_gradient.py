import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torch.autograd import grad as torch_grad
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

def gradient_2_tb(model, writer, name_var, iter_train):

	#gradients = torch_grad(outputs=net_output, inputs=net_input,
#		grad_outputs=torch.ones(net_output.size()).to(device=device).double(),
#		create_graph = True, retain_graph = True, only_inputs=True)[0]

#    gradients_norm = torch.sqrt(torch.sum(gradients**2).item() + 1e-12)
    for i, param in enumerate(model.parameters()):
        gradient = param.grad
        gradient_norm = gradient.norm().item() + 1e-12
        writer.add_scalar('_'.join([name_var, 'layer', str(i)]), gradient_norm, iter_train)
