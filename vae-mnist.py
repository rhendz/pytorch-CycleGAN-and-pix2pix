# This trains a VAE on MNIST dataset for baseline performance

import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import functools
from torchinfo import summary

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from argparse import Namespace

# Set options for VAE
opt = Namespace(batch_size=1, beta1=0.5, checkpoints_dir='./checkpoints', continue_train=False, crop_size=28, 
               dataroot='./datasets/apple2orange_resized_64', dataset_mode='unaligned', direction='AtoB', 
               display_env='main', display_freq=400, display_id=1, display_ncols=4, display_port=8097, 
               display_server='http://localhost', display_winsize=28, epoch='latest', epoch_count=1, 
               gan_mode='lsgan', gpu_ids='0', init_gain=0.02, init_type='normal', input_nc=1, lambda_A=10.0, lambda_B=10.0, 
               lambda_identity=0.5, load_iter=0, load_size=28, lr=0.0002, lr_decay_iters=50, lr_policy='linear', 
               max_dataset_size=np.inf, model='cycle_gan', n_epochs=100, n_epochs_decay=100, n_layers_D=3, 
               name='apple2orange', ndf=64, netD='basic', netG='resnet_3blocks', ngf=64, no_dropout=True, 
               no_flip=False, no_html=False, norm='instance', num_threads=4, output_nc=1, phase='train', 
               pool_size=50, preprocess='resize_and_crop', print_freq=100, save_by_iter=False, 
               save_epoch_freq=5, save_latest_freq=5000, serial_batches=False, suffix='', 
               update_html_freq=1000, use_wandb=False, verbose=False)

# train_loader = create_dataset(opt)
# # print(len(train))
# opt.phase='test'
# test_loader = create_dataset(opt)
# # print(len(test))
# # m = len(dataset)

# # for i_batch, sample_batched in enumerate(dataset):
# #   print(sample_batched)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')