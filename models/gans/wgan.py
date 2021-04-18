# From https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py
import os

import torch
import torch.nn as nn
import torch.optim as optim

from utils import logging

logger = logging.getLogger()


class WGAN(object):
    def __init__(self, channels: int, saving_path: str):
        self.C = channels
        self.G = Generator(self.C)
        self.D = Discriminator(self.C)
        self.saving_path = saving_path

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
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def save_model(self, desc='final_model'):
        path_gen = os.path.join(self.saving_path, f'{desc}_generator.pt')
        path_discr = os.path.join(self.saving_path, f'{desc}_discriminator.pt')
        torch.save(self.G.state_dict(), path_gen)
        torch.save(self.D.state_dict(), path_discr)

    def load_model(self, desc='final_model'):
        path_gen = os.path.join(self.saving_path, f'{desc}_generator.pt')
        path_discr = os.path.join(self.saving_path, f'{desc}_discriminator.pt')
        self.G.load_state_dict(torch.load(path_gen))
        self.D.load_state_dict(torch.load(path_discr))

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
        self.dim = 64
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=self.dim * 4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.dim * 4),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=self.dim * 4, out_channels=self.dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.dim * 2),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=self.dim * 2, out_channels=self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=self.dim),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=self.dim, out_channels=channels, kernel_size=4, stride=2, padding=1))
        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.dim = 64

        self._kernel_size = 4
        self._stride = 2
        self._padding = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels,
                      out_channels=self.dim,
                      kernel_size=self._kernel_size,
                      stride=self._stride,
                      padding=self._padding),
            nn.InstanceNorm2d(self.dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=self.dim,
                      out_channels=self.dim * 2,
                      kernel_size=self._kernel_size,
                      stride=self._stride,
                      padding=self._padding),
            nn.InstanceNorm2d(self.dim * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=self.dim * 2,
                      out_channels=self.dim * 4,
                      kernel_size=self._kernel_size,
                      stride=self._stride,
                      padding=self._padding),
            nn.InstanceNorm2d(self.dim * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=self.dim * 4, out_channels=1, kernel_size=self._kernel_size, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)
