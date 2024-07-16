# %%
# import the libraries
import torch
import pdb
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# %%
# Visualization function
def show(tensor, ch = 1, size = (28, 28), num = 16):
    # tensor: 128 x 784
    data = tensor.detach().cpu().view(-1, ch, *size) # Resize to 28 x 28
    grid = make_grid(data[:num], nrow = 4).permute(1,2,0) # 1 x 28 x 28 =            28 x 28 x 1
                                     # (Original) Index Seq: 0 | 1 |  2  = (Permuted) 1  | 2  | 0
    plt.imshow(grid)
    plt.show()

# Setup of the main parameters and hyperparameters
epochs = 500
current_step = 0
info_step = 300
mean_gen_loss = 0
mean_disc_loss = 0

z_dim = 64
lr = 5e-4
loss_func = nn.BCEWithLogitsLoss()

batch_size = 128
device = 'mps'

dataloader = DataLoader(MNIST('/Users/lurker18/Desktop/GenerativeAI/Data', download = True, transform = transforms.ToTensor()), shuffle = True, batch_size = batch_size)

# Number of Steps = 60_000 / 128 = 468.75 ==> Total Data size / Batch_Size 

# %%
# Declare our models
# Generator
def genBlock(inpu, outpu):
    return nn.Sequential(
        nn.Linear(inpu, outpu),
        nn.BatchNorm1d(outpu),
        nn.ReLU(inplace = True)
    )

class Generator(nn.Module):
    def __init__(self, z_dim = 64, i_dim = 784, h_dim = 128):
        super().__init__()
        self.gen = nn.Sequential(
            genBlock(z_dim,     h_dim),     # 64, 128
            genBlock(h_dim,     h_dim * 2), # 128, 256
            genBlock(h_dim * 2, h_dim * 4), # 256, 512
            genBlock(h_dim * 4, h_dim * 8), # 512, 1024
            nn.Linear(h_dim * 8, i_dim),    # 1024, 784(= 28-height X 28-width)
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.gen(noise)
    
def gen_noise(number, z_dim):
    return torch.randn(number, z_dim).to(device)

## Discriminator
def discBlock(inpu, outpu):
    return nn.Sequential(
        nn.Linear(inpu, outpu),
        nn.LeakyReLU(0.2)
    )

class Discriminator(nn.Module):
    def __init__(self, i_dim = 784, h_dim = 256):
        super().__init__()
        self.disc = nn.Sequential(
            discBlock(i_dim,     h_dim * 4),   # 784, 1024
            discBlock(h_dim * 4, h_dim * 2),   # 1024, 512
            discBlock(h_dim * 2, h_dim),       # 512,  256
            nn.Linear(h_dim, 1)                # 256,   1
        )
    
    def forward(self, image):
        return self.disc(image)


# %%
gen = Generator(z_dim).to(device)
gen_optimize = torch.optim.Adam(gen.parameters(), lr = lr)
disc = Discriminator().to(device)
disc_optimize = torch.optim.Adam(disc.parameters(), lr = lr)

# %%
gen

# %%
disc

# %%
x,y = next(iter(dataloader))
print(x.shape, y.shape)
print(y[:10])

# %%
noise = gen_noise(batch_size, z_dim)
fake = gen(noise)
show(fake)

# %%
# Calculate the Loss
# Generator Loss
def calc_gen_loss(loss_func, gen, disc, number, z_dim):
    noise = gen_noise(number, z_dim)
    fake = gen(noise)
    pred = disc(fake)
    targets = torch.ones_like(pred)
    gen_loss = loss_func(pred, targets)
    return gen_loss

def calc_disc_loss(loss_func, gen, disc, number, real, z_dim):
    noise = gen_noise(number, z_dim)
    fake = gen(noise)
    disc_fake = disc(fake.detach())
    disc_fake_targets = torch.zeros_like(disc_fake)
    disc_fake_loss = loss_func(disc_fake, disc_fake_targets)

    disc_real = disc(real)
    disc_real_targets = torch.ones_like(disc_real)
    disc_real_loss = loss_func(disc_real, disc_real_targets)

    disc_loss = (disc_fake_loss + disc_real_loss) / 2

    return disc_loss

# %%
### 60_000 / 128 = 469 steps in each epoch
### Each step is going to process 128 images = size of the batch_size (except the last step)

for epoch in range(epochs):
    for real, _ in tqdm(dataloader):
        ## Discriminator
        disc_optimize.zero_grad()

        current_batch_size = len(real) # real: 128 x 1ch x 28px x 28px
        real = real.view(current_batch_size, -1) # 128 x 784
        real = real.to(device)

        disc_loss = calc_disc_loss(loss_func, gen, disc, current_batch_size, real, z_dim)

        disc_loss.backward(retain_graph = True)
        disc_optimize.step()
        
        ## Generator
        gen_optimize.zero_grad()
        gen_loss = calc_gen_loss(loss_func, gen, disc, current_batch_size, z_dim)
        gen_loss.backward(retain_graph = True)
        gen_optimize.step()

        # Visuatlization & stats
        mean_disc_loss += disc_loss.item() / info_step
        mean_gen_loss  += gen_loss.item() / info_step

        if current_step % info_step == 0 and current_step > 0:
            fake_noise = gen_noise(current_batch_size, z_dim)
            fake = gen(fake_noise)
            show(fake)
            show(real)
            print(f"{epoch}: Step {current_step} / Gen Loss: {mean_gen_loss} / Disc Loss: {mean_disc_loss}")
            mean_gen_loss, mean_disc_loss = 0,0
        current_step += 1
