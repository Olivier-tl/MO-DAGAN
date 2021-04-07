# From https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py

import torch
import torch.nn as nn
import torch.optim as optim

from utils import logging

logger = logging.getLogger()


class WGAN(object):
    def __init__(self, channels: int, img_shape: int):
        self.img_shape = img_shape
        self.C = channels
        self.G = Generator(self.C)
        self.D = Discriminator(self.C, img_shape)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # Set the logger
        self.number_of_images = 10

        self.critic_iter = 5
        self.lambda_term = 10

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, self.img_shape, self.img_shape))
            else:
                generated_images.append(sample.reshape(self.img_shape, self.img_shape))
        return generated_images

    def save_model(self, path: str):
        torch.save(self.G.state_dict(), f'{path}_generator.pt')
        torch.save(self.D.state_dict(), f'{path}_discriminator.pt')

    def load_model(self, path: str):
        D_model_path = f'{path}_discriminator.pt'
        G_model_path = f'{path}_generator.pt'
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))

    def to(self, device):
        self.G = self.G.to(device)
        self.D = self.D.to(device)
        return self


class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels, img_shape):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.img_shape = img_shape

        self._kernel_size = 4
        self._stride = 2
        self._padding = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels,
                      out_channels=256,
                      kernel_size=self._kernel_size,
                      stride=self._stride,
                      padding=self._padding),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=self._kernel_size,
                      stride=self._stride,
                      padding=self._padding),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512,
                      out_channels=1024,
                      kernel_size=self._kernel_size,
                      stride=self._stride,
                      padding=self._padding),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=self._kernel_size, stride=1, padding=0))

        # conv1_out_dim = (self.img_shape+(2*self._padding)-self._kernel_size)/self._stride+1
        # conv2_out_dim = (conv1_out_dim+(2*self._padding)-self._kernel_size)/self._stride+1

        # self.main_module_out_dim = (conv2_out_dim+(2*self._padding)-self._kernel_size)/self._stride+1
        # self.out_dim = (conv2_out_dim-self._kernel_size)+1

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)
