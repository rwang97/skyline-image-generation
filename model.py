# Importing relevant Libraries
import torch.nn as nn


# Resource on https://github.com/carpedm20/DCGAN-tensorflow
class DCGAN(nn.Module):
    def __init__(self, filter_size=64):
        super(DCGAN, self).__init__()
        self.name = "DC-GAN"
        self.discriminator = Discriminator(filter_size)
        self.generator = Generator(filter_size)
        self.filter_size = filter_size


class Discriminator(nn.Module):
    def __init__(self, filter_size=64):
        super(Discriminator, self).__init__()
        self.name = "Discriminator"
        self.filter_size = filter_size
        self.hidden1 = nn.Sequential(
            nn.Conv2d(3, self.filter_size, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.filter_size),
            nn.LeakyReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(self.filter_size, self.filter_size*2, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.filter_size*2),
            nn.LeakyReLU()
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(self.filter_size * 2, self.filter_size * 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.filter_size * 4),
            nn.LeakyReLU()
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(self.filter_size * 4, self.filter_size * 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.filter_size * 8),
            nn.LeakyReLU()
        )
        # will average all the higher-level features, output 1x1x1
        self.out = nn.Sequential(
            nn.Linear(self.filter_size * 8 * 224 * 224, 1)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.out(x)
        # reshape from [1, 1, 1] to [1] for probability
        x = x.reshape((1))
        return x

# Take sketch of lines as input, and output a generated image
class Generator(nn.Module):
    """
    An eight hidden-layer generative neural network
    """
    def __init__(self, filter_size=64):
        super(Generator, self).__init__()
        self.name = "Generator"
        self.filter_size = filter_size
        self.hidden1 = nn.Sequential(
            nn.Conv2d(1, self.filter_size, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(self.filter_size, self.filter_size * 2, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(self.filter_size * 2, self.filter_size * 4, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(self.filter_size * 4, self.filter_size * 8, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        self.hidden5= nn.Sequential(
            nn.ConvTranspose2d(self.filter_size * 8, self.filter_size * 4, stride = 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hidden6 = nn.Sequential(
            nn.ConvTranspose2d(self.filter_size * 4, self.filter_size * 2, stride = 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hidden7 = nn.Sequential(
            nn.ConvTranspose2d(self.filter_size * 2, self.filter_size * 1, stride=2, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.hidden8 = nn.Sequential(
            nn.ConvTranspose2d(self.filter_size * 1, 3, stride = 2, kernel_size=5, padding=2),
            nn.ReLU()
        )
        # normalize the output from 0 to 1
        self.out = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        x = self.hidden8(x)
        x = self.out(x)
        return x
