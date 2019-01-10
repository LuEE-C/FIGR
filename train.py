"""Training script.
usage: train.py [options]

options:
    --inner_learning_rate=ilr   Learning rate of inner loop [default: 1e-3]
    --outer_learning_rate=olr   Learning rate of outer loop [default: 1e-4]
    --batch_size=bs             Size of task to train with [default: 4]
    --inner_epochs=ie           Amount of meta epochs in the inner loop [default: 10]
    --height=h                  Height of image [default: 32]
    --length=l                  Length of image [default: 32]
    --dataset=ds                Dataset name (Mnist, Omniglot, FIGR8) [default: FIGR8]
    --neural_network=nn         Either ResNet or DCGAN [default: DCGAN]
    -h, --help                  Show this help message and exit
"""
from docopt import docopt


import torch
import torch.optim as optim
import torch.autograd as autograd

from tensorboardX import SummaryWriter
import numpy as np
import os
from environnements import MnistMetaEnv, OmniglotMetaEnv, FIGR8MetaEnv
from model import ResNetDiscriminator, ResNetGenerator, DCGANGenerator, DCGANDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wassertein_loss(inputs, targets):
    return torch.mean(inputs * targets)


def calc_gradient_penalty(discriminator, real_batch, fake_batch):
    epsilon = torch.rand(real_batch.shape[0], 1, device=device)
    interpolates = epsilon.view(-1, 1, 1, 1) * real_batch + (1 - epsilon).view(-1, 1, 1, 1) * fake_batch
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty


def normalize_data(data):
    data *= 2
    data -= 1
    return data


def unnormalize_data(data):
    data += 1
    data /= 2
    return data


class FIGR:
    def __init__(self, args):
        self.load_args(args)
        self.id_string = self.get_id_string()
        self.z_shape = 100
        self.writer = SummaryWriter('Runs/' + self.id_string)
        self.env = eval(self.dataset + 'MetaEnv(height=self.height, length=self.length)')
        self.initialize_gan()
        self.load_checkpoint()

    def inner_loop(self, real_batch):
        self.meta_g.train()
        fake_batch = self.meta_g(torch.tensor(np.random.normal(size=(self.batch_size, self.z_shape)), dtype=torch.float, device=device))
        training_batch = torch.cat([real_batch, fake_batch])

        # Training discriminator
        gradient_penalty = calc_gradient_penalty(self.meta_d, real_batch, fake_batch)
        discriminator_pred = self.meta_d(training_batch)
        discriminator_loss = wassertein_loss(discriminator_pred, self.discriminator_targets)
        discriminator_loss += gradient_penalty

        self.meta_d_optim.zero_grad()
        discriminator_loss.backward()
        self.meta_d_optim.step()

        # Training generator
        output = self.meta_d(self.meta_g(torch.tensor(np.random.normal(size=(self.batch_size, self.z_shape)), dtype=torch.float, device=device)))
        generator_loss = wassertein_loss(output, self.generator_targets)

        self.meta_g_optim.zero_grad()
        generator_loss.backward()
        self.meta_g_optim.step()

        return discriminator_loss.item(), generator_loss.item()

    def validation_run(self):
        data, task = self.env.sample_validation_task(self.batch_size)
        training_images = data.cpu().numpy()
        training_images = np.concatenate([training_images[i] for i in range(self.batch_size)], axis=-1)
        data = normalize_data(data)
        real_batch = data.to(device)

        discriminator_total_loss = 0
        generator_total_loss = 0

        for _ in range(self.inner_epochs):
            disc_loss, gen_loss = self.inner_loop(real_batch)
            discriminator_total_loss += disc_loss
            generator_total_loss += gen_loss

        self.meta_g.eval()
        with torch.no_grad():
            img = self.meta_g(torch.tensor(np.random.normal(size=(self.batch_size * 3, self.z_shape)), dtype=torch.float, device=device))
        img = img.detach().cpu().numpy()
        img = np.concatenate([np.concatenate([img[i * 3 + j] for j in range(3)], axis=-2) for i in range(self.batch_size)], axis=-1)
        img = unnormalize_data(img)
        img = np.concatenate([training_images, img], axis=-2)
        self.writer.add_image('Validation_generated', img, self.eps)
        self.writer.add_scalar('Validation_discriminator_loss', discriminator_total_loss, self.eps)
        self.writer.add_scalar('Validation_generator_loss', generator_total_loss, self.eps)

    def meta_training_loop(self):
        data, task = self.env.sample_training_task(self.batch_size)
        data = normalize_data(data)
        real_batch = data.to(device)

        discriminator_total_loss = 0
        generator_total_loss = 0

        for _ in range(self.inner_epochs):
            disc_loss, gen_loss = self.inner_loop(real_batch)
            discriminator_total_loss += disc_loss
            generator_total_loss += gen_loss

        self.writer.add_scalar('Training_discriminator_loss', discriminator_total_loss, self.eps)
        self.writer.add_scalar('Training_generator_loss', generator_total_loss, self.eps)

        # Updating both generator and dicriminator
        for p, meta_p in zip(self.g.parameters(), self.meta_g.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.g_optim.step()

        for p, meta_p in zip(self.d.parameters(), self.meta_d.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.d_optim.step()

    def reset_meta_model(self):
        self.meta_g.train()
        self.meta_d.train()
        self.meta_d.load_state_dict(self.d.state_dict())
        self.meta_g.load_state_dict(self.g.state_dict())

    def training(self):
        while self.eps <= 1000000:
            self.reset_meta_model()
            self.meta_training_loop()

            # Validation run every 10000 training loop
            if self.eps % 10000 == 0:
                self.reset_meta_model()
                self.validation_run()
                self.checkpoint_model()
            self.eps += 1


    def load_args(self, args):
        self.outer_learning_rate = float(args['--outer_learning_rate'])
        self.inner_learning_rate = float(args['--inner_learning_rate'])
        self.batch_size = int(args['--batch_size'])
        self.inner_epochs = int(args['--inner_epochs'])
        self.height = int(args['--height'])
        self.length = int(args['--length'])
        self.dataset = args['--dataset']
        self.neural_network = args['--neural_network']

    def load_checkpoint(self):
        if os.path.isfile('Runs/' + self.id_string + '/checkpoint'):
            checkpoint = torch.load('Runs/' + self.id_string + '/checkpoint')
            self.d.load_state_dict(checkpoint['discriminator'])
            self.g.load_state_dict(checkpoint['generator'])
            self.eps = checkpoint['episode']
        else:
            self.eps = 0

    def get_id_string(self):
        return '{}_{}_olr{}_ilr{}_bsize{}_ie{}_h{}_l{}'.format(self.neural_network,
                                                                         self.dataset,
                                                                         str(self.outer_learning_rate),
                                                                         str(self.inner_learning_rate),
                                                                         str(self.batch_size),
                                                                         str(self.inner_epochs),
                                                                         str(self.height),
                                                                         str(self.length))

    def initialize_gan(self):
        # D and G on CPU since they never do a feed forward operation
        self.d = eval(self.neural_network + 'Discriminator(self.env.channels, self.env.height, self.env.length)')
        self.g = eval(self.neural_network + 'Generator(self.z_shape, self.env.channels, self.env.height, self.env.length)')
        self.meta_d = eval(self.neural_network + 'Discriminator(self.env.channels, self.env.height, self.env.length)').to(device)
        self.meta_g = eval(self.neural_network + 'Generator(self.z_shape, self.env.channels, self.env.height, self.env.length)').to(device)
        self.d_optim = optim.Adam(params=self.d.parameters(), lr=self.outer_learning_rate)
        self.g_optim = optim.Adam(params=self.g.parameters(), lr=self.outer_learning_rate)
        self.meta_d_optim = optim.SGD(params=self.meta_d.parameters(), lr=self.inner_learning_rate)
        self.meta_g_optim = optim.SGD(params=self.meta_g.parameters(), lr=self.inner_learning_rate)

        self.discriminator_targets = torch.tensor([1] * self.batch_size + [-1] * self.batch_size, dtype=torch.float, device=device).view(-1, 1)
        self.generator_targets = torch.tensor([1] * self.batch_size, dtype=torch.float, device=device).view(-1, 1)


    def checkpoint_model(self):
        checkpoint = {'discriminator': self.d.state_dict(),
                      'generator': self.g.state_dict(),
                      'episode': self.eps}
        torch.save(checkpoint, 'Runs/' + self.id_string + '/checkpoint')

if __name__ == '__main__':
    args = docopt(__doc__)
    env = FIGR(args)
    env.training()
