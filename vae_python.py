import os
print(os.getcwd())

# Run this line when initially downloading the datasets.
# !bash ./datasets/download_cyclegan_dataset.sh apple2orange

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


# # Load in the data using cycle gan's data loader

# opt = Namespace(batch_size=1, beta1=0.5, checkpoints_dir='./checkpoints', continue_train=False, crop_size=256, 
#                dataroot='./datasets/apple2orange', dataset_mode='unaligned', direction='AtoB', 
#                display_env='main', display_freq=400, display_id=1, display_ncols=4, display_port=8097, 
#                display_server='http://localhost', display_winsize=256, epoch='latest', epoch_count=1, 
#                gan_mode='lsgan', gpu_ids='0', init_gain=0.02, init_type='normal', input_nc=1, lambda_A=10.0, lambda_B=10.0, 
#                lambda_identity=0.5, load_iter=0, load_size=256, lr=0.0002, lr_decay_iters=50, lr_policy='linear', 
#                max_dataset_size=np.inf, model='cycle_gan', n_epochs=100, n_epochs_decay=100, n_layers_D=3, 
#                name='apple2orange', ndf=64, netD='basic', netG='resnet_3blocks', ngf=64, no_dropout=True, 
#                no_flip=False, no_html=False, norm='instance', num_threads=4, output_nc=1, phase='train', 
#                pool_size=50, preprocess='resize_and_crop', print_freq=100, save_by_iter=False, 
#                save_epoch_freq=5, save_latest_freq=5000, serial_batches=False, suffix='', 
#                update_html_freq=1000, use_wandb=False, verbose=False)


# # Load in the train dataset
# train_loader = create_dataset(opt)

# # Load in the test dataset
# opt.phase='test'
# test_loader = create_dataset(opt)

# for data in train_loader:
#     print(data['A'].size())
#     break

# Test VAE with Mnist dataset

data_dir = 'dataset'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

print(train_dataset)

train_transform = transforms.Compose([
transforms.ToTensor(),
])

test_transform = transforms.Compose([
transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=32

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)




from models.networks import *

shp = 3
mnist_padding = 1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):  
        super(VariationalEncoder, self).__init__()
        nc = 1 # num channels
        self.conv1 = nn.Conv2d(nc, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)  
        self.linear1 = nn.Linear(32*shp*shp, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, shp * shp * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

# Serv
class Decoder2(nn.Module):
  def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, shp * shp * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, shp, shp))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=mnist_padding, output_padding=1)
        )

        self.generator = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

  def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = self.generator(x)
        x = torch.sigmoid(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder2(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)

# ### Training function
# def train_epoch(vae, device, dataloader, optimizer):
#     # Set train mode for both the encoder and the decoder
#     vae.train()
#     train_loss = 0.0
#     # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
#     for x in dataloader: 
#         # print(x)
#         # Move tensor to the proper device
#         x['A'] = x['A'].to(device)
#         x_hat = vae(x['A'])
#         # Evaluate loss
#         loss = ((x['A'] - x_hat)**2).sum() + vae.encoder.kl

#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # Print batch loss
#         # print('\t partial train loss (single batch): %f' % (loss.item()))
#         train_loss+=loss.item()

#     return train_loss / len(dataloader.dataset)

# ### Testing function
# def test_epoch(vae, device, dataloader):
#     # Set evaluation mode for encoder and decoder
#     vae.eval()
#     val_loss = 0.0
#     with torch.no_grad(): # No need to track the gradients
#         for x in dataloader:
#             # Move tensor to the proper device
#             x['A'] = x['A'].to(device)
#             # Encode data
#             encoded_data = vae.encoder(x['A'])
#             # Decode data
#             x_hat = vae(x['A'])
#             loss = ((x['A'] - x_hat)**2).sum() + vae.encoder.kl
#             val_loss += loss.item()

#     return val_loss / len(dataloader.dataset)

### Set the random seed for reproducible results
torch.manual_seed(0)

num_epochs = 100
d = 4

vae = VariationalAutoencoder(latent_dims=d)

lr = 1e-3 

optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

vae.to(device)

### Training function
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader: 
        # print(x)
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)

### Testing function
def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)

# enc = VariationalEncoder(4)
# summary(enc, input_size=(256,1,32,32))

# dec = Decoder2(4)
# summary(dec, input_size=(256,4))

vae = VariationalAutoencoder(16)
summary(vae, input_size=(8,1,28,28))

def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()  


# Toggle this if not using MNIST
# valid_loader = test_loader

vae_encoder = 0
vae_decoder = 0

for epoch in range(num_epochs):
  train_loss = train_epoch(vae,device,train_loader,optim)
  val_loss = test_epoch(vae,device,valid_loader)
  print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
  # vae_encoder = vae.encoder
  # vae_decoder = vae.decoder
  plot_ae_outputs(vae.encoder,vae.decoder,n=10)