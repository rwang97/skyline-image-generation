import torch.nn as nn

class Discriminator(nn.Module):
    """
    A six hidden-layer discriminative neural network
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hidden5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hidden6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # will average all the higher-level features, output 1x1x1
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool3d(1)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.out(x)
        # reshape from [1, 1, 1] to [1] for probability
        x = x.reshape((1))
        return x

# Take sketch of lines as input, and output a generated image
class Generator(nn.Module):
    """
    An eight hidden-layer generative neural network
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hidden5 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hidden6 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.hidden7 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hidden8 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),
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
