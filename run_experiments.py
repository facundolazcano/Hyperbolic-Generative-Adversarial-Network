import numpy as np
import subprocess
import itertools
import os
import os.path as op
import sys
import pandas as pd
import argparse
import json

# Constants
root_path = '/home/jenny2/Experiments/'

# templates
name = 'D_%s_G_%s_cd_%s_cg_%s_epochs_%i_trainc_%i_lrmc_%s'
cmd = 'python %s --root %s --name %s --cfg_d %s --cfg_g %s --dataset MNIST --c_d %f --c_g %f --mode_train_c_g %i --mode_train_c_d %i --lrmul_c %f --epochs %i --device %i'

# tables
archs_config = {
    0: 'train.py',
    1: 'train_cgan.py',
    2: 'train_wgan.py'
}
root_config = {
    0: 'HGAN',
    1: 'HCGAN',
    2: 'HWGAN'
}


# Run main algorithm
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Runner Experiments')
    
    # arguments
    parser.add_argument('--config', type=str, help='path of config json file')
    parser.add_argument('--device', type=int, default=2, help='GPU device')
    parser.add_argument('--arch', type=int, help='0: hgan, 1:hcgan, 2:hwgan')
    
    args = parser.parse_args()

    # get arch
    arch = archs_config[args.arch]
    root_path_exp = op.join(root_path, root_config[args.arch])
    
    # read config of experiments
    with open(args.config) as json_file:
         data = json.load(json_file)
    experiments = data.get('experiments')
    date = data.get('date')
    print('experiments:')
    print(root_path_exp)
    print(experiments)
    print(date)

    # autorizacion experiments
    decision = ''
    while decision not in ['n', 'y']:
        decision = input('¿continuar?(y/n)')
    if decision == 'n':
        sys.exit(0)

    # run experiments
    for exp_detail in experiments:
        name_arch = name % (exp_detail['dis'], exp_detail['gen'],
                    np.format_float_scientific(exp_detail['c_d'], exp_digits=2),
                    np.format_float_scientific(exp_detail['c_g'], exp_digits=2),
                    exp_detail['epochs'], exp_detail['mode_train_c'],
                    np.format_float_scientific(exp_detail['lrmul_c'], exp_digits=2))

        run = cmd % (arch, root_path_exp, name_arch,
                     exp_detail['dis'], exp_detail['gen'], 
                     exp_detail['c_d'], exp_detail['c_g'],
                     exp_detail['mode_train_c'], exp_detail['mode_train_c'],
                     exp_detail['lrmul_c'],
                     exp_detail['epochs'], args.device)
        print(run)
        
        subprocess.run(run.split())





