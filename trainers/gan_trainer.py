# Heavily inspired from https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py

import os
from typing import List

import wandb
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import utils
from tqdm import tqdm
from PIL import Image

from utils import logging, Config, AugmentPipe
from .trainer import Trainer

logger = logging.getLogger()

IMG_SAMPLES_PATH = 'output/gan_samples'
SAVE_PER_TIMES = 1000
ADA_UPDATE_INTERVAL = 4
ADA_TARGET = 0.6
ADA_IMG_ZERO_ONE = 20000


class GANTrainer(Trainer):
    def __init__(self, trainer_config: Config.Trainer, model: torch.nn.Module, dataset: DataLoader, classes: List[int]):
        super(GANTrainer, self).__init__(model, classes)
        self.dataset = dataset
        self.d_optimizer = self._get_optimizer(trainer_config.optimizer, model.D, trainer_config.lr,
                                               trainer_config.betas)
        self.g_optimizer = self._get_optimizer(trainer_config.optimizer, model.G, trainer_config.lr,
                                               trainer_config.betas)
        self.ada = trainer_config.ada
        self.generator_iters = trainer_config.epochs
        self.augment_pipe = None
        if self.ada:
            self.augment_pipe = AugmentPipe().to(self.device)
            self.augment_pipe.p = torch.Tensor([0]).to(self.device)

    def train(self):

        one = torch.tensor(1, dtype=torch.float).to(self.device)
        mone = (one * -1).to(self.device)
        ada_stats = []  # Discriminator logits sign

        for g_iter in tqdm(range(self.generator_iters), desc='Generator Iterations'):

            for p in self.model.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0

            for d_iter in range(self.model.critic_iter):
                self.model.D.zero_grad()

                images = next(iter(self.dataset))[0].to(self.device)
                if self.ada:
                    images = self.augment_pipe.forward(images)

                # Check for batch to have full batch_size
                if (images.size()[0] != self.dataset.batch_size):
                    continue

                # ---------------------
                # Train discriminator
                # ---------------------

                # Train with real images
                d_loss_real = self.model.D(images)

                if self.ada:
                    ada_stats.append(d_loss_real.detach().flatten())
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = torch.randn(self.dataset.batch_size, 100, 1, 1).to(self.device)

                fake_images = self.model.G(z)
                if self.ada:
                    fake_images = self.augment_pipe.forward(fake_images)

                d_loss_fake = self.model.D(fake_images)

                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self._calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()

                # Log to WandB
                overall_iter = g_iter * self.model.critic_iter + d_iter
                wandb.log({'d_loss_fake': d_loss_fake, 'd_loss_real': d_loss_real, 'd_iter': overall_iter})
                
            # ---------------------
            # Train generator
            # ---------------------

            # Generator update
            for p in self.model.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.model.G.zero_grad()

            # compute loss with fake images
            z = torch.randn(self.dataset.batch_size, 100, 1, 1).to(self.device)
            fake_images = self.model.G(z)
            if self.ada:
                fake_images = self.augment_pipe.forward(fake_images)
            g_loss = self.model.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()

            # Update augment strength
            if self.ada and (g_iter % ADA_UPDATE_INTERVAL == 0) and len(ada_stats) != 0:
                r_t = torch.mean(torch.sign(torch.stack(ada_stats)))
                adjust = torch.sign(r_t - ADA_TARGET) * (self.dataset.batch_size *
                                                         ADA_UPDATE_INTERVAL) / ADA_IMG_ZERO_ONE
                self.augment_pipe.p = (self.augment_pipe.p + adjust).clamp(0)
                ada_stats = []
                wandb.log({'ada/p': self.augment_pipe.p.cpu(), 'ada/r_t': r_t.cpu(), 'ada/p_step': adjust.cpu()})

            # Log to WandB
            wandb.log({'g_loss': g_loss, 'g_iter': g_iter})

            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                #self.save_model(desc=f'iter_{g_iter}')

                if not os.path.exists(IMG_SAMPLES_PATH):
                    os.makedirs(IMG_SAMPLES_PATH)

                # Denormalize images and save them in grid 8x8
                z = torch.randn(800, 100, 1, 1).to(self.device)
                samples = self.model.G(z)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()[:64]
                grid = utils.make_grid(samples)
                utils.save_image(grid, os.path.join(IMG_SAMPLES_PATH, f'img_generator_iter_{g_iter}.png'))

                # Log to WandB
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                wandb.log({'g_sample': [wandb.Image(im, caption=f'g_iter_{g_iter}')]})

        # All done. Save the trained parameters
        self.save_model(desc='final_model')

    def test(self):
        # self.load_model(D_model_path, G_model_path)
        z = torch.randn(self.dataset.batch_size, 100, 1, 1).to(self.device)
        samples = self.model.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()

        # Save samples
        grid = utils.make_grid(samples)
        utils.save_image(grid, 'final_gan_model_image.png')

    def _calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.dataset.batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.dataset.batch_size, real_images.size(1), real_images.size(2),
                         real_images.size(3)).to(self.device)
        interpolated = eta * real_images + ((1 - eta) * fake_images).to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.model.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated,
                                        inputs=interpolated,
                                        grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                                        create_graph=True,
                                        retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * self.model.lambda_term
        return grad_penalty