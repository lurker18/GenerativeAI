# %%
import torch
import torchvision
import os
import PIL
import pdb

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import wandb
wandb.login()

# %%
def show(tensor, num = 25, wandbactive = 0, name = ''):
    data = tensor.detach().cpu()
    grid = make_grid(data[:num], nrow = 5).permute(1,2,0)

    if (wandbactive == 1 and wandbact == 1):
        wandbactive.log({name:wandbactive.Image(grid.numpy().clip(0, 1))})

    plt.imshow(grid.clip(0,1))
    plt.show()

### hyperparameters and general parameters
n_epochs = 10_000
batch_size = 128
lr = 1e-4
z_dim = 200
device = 'mps'

current_step = 0
critic_cycles = 5
gen_losses = []
critic_losses = []
show_step = 35
save_step = 35

wandbact = 1 # OPTIONAL: yes, we want to track stats through weights and biases

# %%
%%capture
experiment_name = wandb.util.generate_id()

myrun = wandb.init(
    project = 'wgen',
    group = experiment_name,
    config = {
        "optimizer" : "adam",
        "model" : "wgen gp",
        "epoch" : "1000",
        "batch_size": 128
    }
)
config = wandb.config
print(experiment_name)
# %%
print(experiment_name)

# %%
# Generate model

class Generator(nn.Module):
    def __init__(self, z_dim = 64, d_dim = 16):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, d_dim * 32, 4, 1, 0), ## (initial): 1 x 1 (ch: 200) img ==> 4 x 4 (ch: 512) img 
            nn.BatchNorm2d(d_dim * 32),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 32, d_dim * 16, 4, 2, 1), ## (2nd layer): 4 x 4 (ch: 512) img ==> 8 x 8 (ch:256) img
            nn.BatchNorm2d(d_dim * 16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d_dim * 16, d_dim * 8, 4, 2, 1), ## (3rd layer): 8 x 8 (ch: 256) img ==> 16 x 16 (ch:128) img
            nn.BatchNorm2d(d_dim * 8),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(d_dim * 8, d_dim * 4, 4, 2, 1), ## (4th layer): 16 x 16 (ch: 128) img ==> 32 x 32 (ch:64) img
            nn.BatchNorm2d(d_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 4, d_dim * 2, 4, 2, 1), ## (5th layer): 32 x 32 (ch: 64) img ==> 64 x 64 (ch:32) img
            nn.BatchNorm2d(d_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 2, 3, 4, 2, 1), ## (Last layer): 64 x 64 (ch: 32) img ==> 128 x 128 (ch:3) img
            nn.Tanh() # Produce result in the range from -1 to 1
        )
    
    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1) # 128 x 200 x 1 x 1
        return self.gen(x)
    
def gen_noise(num, z_dim, device = 'mps'):
    return torch.randn(num, z_dim, device = device) # 128 x 200

# %%
# Critic Model
# Conv2d: in_channels, out_channels, kernel_size, stride = 1, padding = 0
## New width and height: # (n+2*pad-ks) // stride +1

class Critic(nn.Module):
    def __init__(self, d_dim = 16):
        super(Critic, self).__init__()

        self.critics = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = d_dim, kernel_size = 4, stride = 2, padding = 1), # (128+2*1-4)//2+1 = 64 x 64 (ch: 3 => 16)
            nn.InstanceNorm2d(d_dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = d_dim, out_channels = d_dim * 2, kernel_size = 4, stride = 2, padding = 1), # 32 x 32 (ch: 16 => 32)
            nn.InstanceNorm2d(d_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = d_dim * 2, out_channels = d_dim * 4, kernel_size = 4, stride = 2, padding = 1), # 16 x 16 (ch: 32 => 64)
            nn.InstanceNorm2d(d_dim * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = d_dim * 4, out_channels = d_dim * 8, kernel_size = 4, stride = 2, padding = 1), # 8 x 8 (ch: 64 => 128)
            nn.InstanceNorm2d(d_dim * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = d_dim * 8, out_channels = d_dim * 16, kernel_size = 4, stride = 2, padding = 1), # 4 x 4 (ch: 128 => 256)
            nn.InstanceNorm2d(d_dim * 16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = d_dim * 16, out_channels = 1, kernel_size = 4, stride = 1, padding = 9), # 1 x 1 (ch: 256 => 1)
        )
    
    def forward(self, image):
        # image : 128 x 3 x 128 x 128 (BS: CH: HEIGHT: WIDTH)
        critic_pred = self.critics(image) # 128 x 1 x 1 x 1
        return critic_pred.view(len(critic_pred), -1) ## 128 x 1
# %%
# OPTIONAL: Init your weights in different ways
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
    
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

    
# %%
#gen = gen.apply(init_weights)
#crit = crit.apply(init_weights)

