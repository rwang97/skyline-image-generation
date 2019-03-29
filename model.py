# Importing relevant Libraries
import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGAN(nn.Module):
    def __init__(self, device, filter_size=64, num_channel=3, color='R', ngpu=0, cloud_computing=False):
        super(DCGAN, self).__init__()
        self.name = "DC-GAN"
        self.color = color
        self.netD = Discriminator(filter_size=filter_size, num_channel=num_channel, ngpu=ngpu)
        self.netG = Generator(num_downsampling=8, filter_size=filter_size, num_channel=num_channel, ngpu=ngpu)
        if cloud_computing == True:
            self.netD = self.netD.to(device)
            self.netG = self.netG.to(device)
            if (device.type == 'cuda') and (ngpu > 1):
                self.netG = nn.DataParallel(self.netG, list(range(ngpu)))
                self.netD = nn.DataParallel(self.netD, list(range(ngpu)))
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

class Discriminator(nn.Module):
    def __init__(self, filter_size=64, num_channel=3, ngpu=0):
        super(Discriminator, self).__init__()
        self.name = "Discriminator"
        self.filter_size = filter_size
        self.ngpu = ngpu
        # 256 -> 128
        self.layer1 = nn.Sequential(
            nn.Conv2d(2*num_channel, self.filter_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 128 -> 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.filter_size, self.filter_size*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.filter_size*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64 -> 32
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.filter_size * 2, self.filter_size * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.filter_size * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32 -> 31
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.filter_size * 4, self.filter_size * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(self.filter_size * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 31 -> 30
        self.layer5 = nn.Sequential(
            nn.Conv2d(self.filter_size * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

# Take sketch of lines as input, and output a generated image
class Generator(nn.Module):
    """
    An eight hidden-layer generative neural network
    """
    def __init__(self, num_downsampling=8, filter_size=64, num_channel=3, ngpu=0):
        super(Generator, self).__init__()
        self.name = "Generator"
        self.ngpu = ngpu
        self.model = Unet(num_downsampling=num_downsampling, filter_size=filter_size, num_channel=num_channel)

    def forward(self, x):
        x = self.model(x)
        return x

# Recursive Unet implementation from
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class Unet(nn.Module):
    def __init__(self, num_downsampling=8, filter_size=64, num_channel=3):
        super(Unet, self).__init__()
        # innermost layer
        unet_block = UnetBlock(filter_size * 8, filter_size * 8, None, innermost=True)
        # intermediate layers
        for i in range(num_downsampling - 5):
            unet_block = UnetBlock(filter_size * 8, filter_size * 8, None, unet_block, dropout=True)
        # downsampling and upsampling layers
        unet_block = UnetBlock(filter_size * 8, filter_size * 4, None, unet_block)
        unet_block = UnetBlock(filter_size * 4, filter_size * 2, None, unet_block)
        unet_block = UnetBlock(filter_size * 2, filter_size, None, unet_block)
        # outermost layer
        self.model = UnetBlock(filter_size, num_channel, num_channel, unet_block, outermost=True)

    def forward(self, input):
        return self.model(input)

# Unet Blocks that uses skip connection
class UnetBlock(nn.Module):
    # in_channel and out_channel are naming from transposed convolution side
    def __init__(self, in_channel, out_channel, input_channel=None, subnet=None, outermost=False, innermost=False, dropout=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if input_channel is None:
            input_channel = out_channel
        downconv = nn.Conv2d(input_channel, in_channel, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = nn.InstanceNorm2d(in_channel)
        uprelu = nn.ReLU(inplace=True)
        upnorm = nn.InstanceNorm2d(out_channel)

        if innermost:
            upconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        elif outermost:
            upconv = nn.ConvTranspose2d(in_channel * 2, out_channel, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [subnet] + up
        else:
            upconv = nn.ConvTranspose2d(in_channel * 2, out_channel, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if dropout:
                model = down + [subnet] + up + [nn.Dropout(0.5)]
            else:
                model = down + [subnet] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
