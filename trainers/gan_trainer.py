# Heavily inspired from https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py

import torch
from torch.utils.data import DataLoader
from torchvision import utils

from utils import logging
from .trainer import Trainer

logger = logging.getLogger()

IMG_SAMPLES_PATH = 'output/gan_samples'


class GANTrainer(Trainer):
    def __init__(self,
                 model: torch.nn.Module,
                 dataset: DataLoader,
                 lr: float = 0.001,
                 optimizer: str = "adam",
                 loss: str = "cross_entropy"):
        super(GANTrainer, self).__init__(model, lr, optimizer, loss)
        self.dataset = dataset

    def train(self):

        one = torch.tensor(1, dtype=torch.float).to(self.device)
        mone = (one * -1).to(self.device)

        for g_iter in range(self.model.generator_iters):

            for p in self.model.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0

            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.dataset.__next__().to(self.device)

                # Check for batch to have full batch_size
                if (images.size()[0] != self.dataset.batch_size):
                    continue

                # ---------------------
                # Train discriminator
                # ---------------------

                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = torch.randn(self.dataset.batch_size, 100, 1, 1).to(self.device)

                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self._calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                logger.info(
                    f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}'
                )

            # ---------------------
            # Train generator
            # ---------------------

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()

            # compute loss with fake images
            z = torch.randn(batch_size, 100, 1, 1)
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            logger.info(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')

            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model(epoch=g_iter)

                if not os.path.exists(IMG_SAMPLES_PATH):
                    os.makedirs(IMG_SAMPLES_PATH)

                # Denormalize images and save them in grid 8x8
                z = self.get_torch_variable(torch.randn(800, 100, 1, 1))
                samples = self.G(z)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()[:64]
                grid = utils.make_grid(samples)
                utils.save_image(grid, os.path.join(IMG_SAMPLES_PATH, f'img_generator_iter_{g_iter}.png'))

                #
                # TODO: Add WandB logging
                #

        # All done. Save the trained parameters
        self.save_model(epoch=g_iter)

    def test(self):
        # self.load_model(D_model_path, G_model_path)
        z = torch.randn(self.batch_size, 100, 1, 1).to(self.device)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()

        # Save samples
        grid = utils.make_grid(samples)
        utils.save_image(grid, 'final_gan_model_image.png')

    def _calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3)).to(self.device)
        interpolated = eta * real_images + ((1 - eta) * fake_images).to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated,
                                  inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).cuda(self.cuda_index)
                                  if self.cuda else torch.ones(prob_interpolated.size()),
                                  create_graph=True,
                                  retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * self.lambda_term
        return grad_penalty
