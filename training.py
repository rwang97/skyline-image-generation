import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from model import DCGAN
import cv2 as cv
import shutil

manualSeed = 999

###############################################################################
# Data Loading
def get_data_loader(batch_size):
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    transform_real = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_edge = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # load each dataset with corresponding folders
    realset = torchvision.datasets.ImageFolder(root='./data', transform=transform_real)
    real_loader = torch.utils.data.DataLoader(realset, batch_size=batch_size,
                                               num_workers=1)

    # load each dataset with corresponding folders
    edgeset = torchvision.datasets.ImageFolder(root='./generator_input', transform=transform_edge)
    edge_loader = torch.utils.data.DataLoader(edgeset, batch_size=batch_size,
                                               num_workers=1)

    ############# plotting images ############
    # edge_iter = iter(edge_loader)
    # k = 0
    # for image, labels in real_loader:
    #     sample_batch = next(edge_iter)
    #     image_1 = image[0]
    #     image_2 = sample_batch[0][0]
    #     img_1 = np.transpose(image_1, [1, 2, 0])
    #     img_2 = np.transpose(image_2, [1, 2, 0])
    #     img_1 = img_1 / 2 + 0.5
    #     img_2 = img_2 / 2 + 0.5
    #
    #     plt.subplot(4, 2, k + 1)
    #     plt.imshow(img_1)
    #     plt.subplot(4, 2, k + 2)
    #     plt.imshow(img_2)
    #     k += 2
    #     if k > 12:
    #         input("Press Enter to continue...")
    #         break

    return real_loader, edge_loader


###############################################################################
# Weight initialization
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def train(model, batch_size=32, learning_rate=1e-4, num_epochs=5):
    # clean the fake directory
    if os.path.exists('./data/Fake'):
        shutil.rmtree('./data/Fake')
    os.makedirs('./data/Fake')
    
    # load training data
    real_loader, edge_loader = get_data_loader(batch_size)

    # loss function and optimizer
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(model.netD.parameters(), lr=learning_rate)
    optimizerG = optim.Adam(model.netG.parameters(), lr=learning_rate)

    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        edge_iter = iter(edge_loader)
        # For each batch in the dataloader
        transform_edge = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])

        test_edge = torchvision.datasets.ImageFolder(root='./test', transform=transform_edge)
        test_loader = torch.utils.data.DataLoader(test_edge, batch_size=1,num_workers=1)
        test_fake = model.netG(next(iter(test_loader))[0])
        test_fake = np.transpose(test_fake.detach().numpy().squeeze(), [1, 2, 0]) * 255
        # print(test_fake.shape)
        # print(test_fake)
        cv.imwrite("./data/Fake/" + str(epoch) + ".jpg", test_fake)

        for i, real_data in enumerate(real_loader, 0):

            ############################
            model.netD.zero_grad()
            # Format batch
            label = torch.full((batch_size,), real_label)
            # Forward pass real batch through D
            output = model.netD(real_data[0]).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate fake image batch with G
            fake_batch = next(edge_iter)
            fake = model.netG(fake_batch[0])
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = model.netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            model.netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = model.netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            print("epoch: " + str(epoch) + ", iteration: " + str(i) + ", d_loss is: " + str(float(errD)) + ", g_loss is: " + str(float(errG)))
            print("=============================================================")


###############################################################################
# Main Function
if __name__ == '__main__':
    filter_size = 64
    gan = DCGAN(filter_size)
    gan.netG.apply(weights_init)
    gan.netD.apply(weights_init)
    num_epoch = 5
    batch_size = 32
    learning_rate = 1e-4
    train(gan, batch_size, learning_rate, num_epoch)
    #
    # print images
    #real_loader, edge_loader = get_data_loader(1)
    #train_model(gan)