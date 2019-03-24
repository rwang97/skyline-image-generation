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

# =================================== Load Data ======================================
def get_data_loader(num_channel, batch_size):
    # We transform them to Tensors of normalized range [-1, 1].
    if num_channel == 1:
        real_dir = './input_edges'
        input_dir = './mor_edges'
        test_dir = './test'
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        real_dir = './data'
        input_dir = './denoise'
        test_dir = './test'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    real_set = torchvision.datasets.ImageFolder(root=real_dir, transform=transform) # real images
    edge_set = torchvision.datasets.ImageFolder(root=input_dir, transform=transform) # input images for generator

    np.random.seed(1000)  # Fixed numpy random seed for reproducible shuffling
    train_indices = np.arange(len(real_set))
    np.random.shuffle(train_indices)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    real_loader = torch.utils.data.DataLoader(real_set, batch_size=batch_size, sampler=train_sampler, num_workers=1, drop_last=True)
    edge_loader = torch.utils.data.DataLoader(edge_set, batch_size=batch_size, sampler=train_sampler, num_workers=1, drop_last=True)

    # test image
    test_edge = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_edge, batch_size=1, num_workers=1)

    return real_loader, edge_loader, test_loader

# ============================= Weight Initialization ======================================
# Weight initialization
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def test_output(model, test_loader, num_channel, epoch):
    test_fake = model.netG(next(iter(test_loader))[0])
    if num_channel == 1:
        test_fake = test_fake.detach().numpy().squeeze()
    else:
        test_fake = np.transpose(test_fake.detach().numpy().squeeze(), [1, 2, 0])

    test_fake = (test_fake / 2 + 0.5) * 255
    cv.imwrite("./data/Fake/" + str(epoch) + ".jpg", test_fake)

# =================================== Checkpoint ======================================
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "./checkpoints/model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

# =================================== Training ======================================
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def train(model, num_channel=1, batch_size=32, learning_rate=1e-4, L1_lambda=10, num_epochs=5, checkpoint=False):

    if checkpoint:
        if os.path.exists('./checkpoints'):
            shutil.rmtree('./checkpoints')
        os.makedirs('./checkpoints')

    # clean the fake directory
    if os.path.exists('./data/Fake'):
        shutil.rmtree('./data/Fake')
    os.makedirs('./data/Fake')
    
    # load training data
    real_loader, edge_loader, test_loader = get_data_loader(num_channel, batch_size)

    # loss function and optimizer
    BCE_Loss = nn.BCELoss()
    L1_Loss = nn.L1Loss()
    optimizerD = optim.Adam(model.netD.parameters(), lr=learning_rate)
    optimizerG = optim.Adam(model.netG.parameters(), lr=learning_rate)

    real_label = 1
    fake_label = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        edge_iter = iter(edge_loader)

        # output result to disk
        test_output(model, test_loader, num_channel, epoch)

        for i, real_data in enumerate(real_loader, 0):
            ############################
            # (1) Update D network
            ############################
            model.netD.zero_grad()
            # Train with all-fake batch
            # Generate fake image batch with G
            edge_batch = next(edge_iter)
            fake = model.netG(edge_batch[0])

            # Forward pass real batch through D
            output = model.netD(edge_batch[0], real_data[0])
            label = torch.full(output.shape, real_label)
            # Calculate loss on all-real batch
            loss_D_real = BCE_Loss(output, label)
            D_real = output.mean().item()

            # Classify all fake batch with D
            output = model.netD(edge_batch[0], fake.detach())
            label = torch.full(output.shape, fake_label)
            # Calculate D's loss on the all-fake batch
            loss_D_fake = BCE_Loss(output, label)
            D_fake = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            model.netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = model.netD(edge_batch[0], fake)
            # Calculate G's loss based on this output
            loss_G_BCE = BCE_Loss(output, label)
            loss_G_L1 = L1_Loss(fake, real_data[0]) * L1_lambda
            loss_G = loss_G_BCE + loss_G_L1
            # Calculate gradients for G
            loss_G.backward()
            optimizerG.step()

            # Output training stats
            print("epoch: " + str(epoch) + ", iteration: " + str(i))
            print("d_loss is: " + str(float(loss_D)) + ", g_loss is: " + str(float(loss_G)))
            print("d_real is: " + str(D_real) + ", d_fake is: " + str(D_fake))
            print("================================================================")

        if checkpoint:
            # Save the current model (checkpoint) to a file
            model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
            torch.save(model.state_dict(), model_path)

    # output final result to disk
    test_output(model, test_loader, num_channel, epoch)


# =================================== Main ======================================
if __name__ == '__main__':
    filter_size = 64
    num_channel = 3
    num_epoch = 15
    batch_size = 128
    learning_rate = 1e-3
    L1_lambda = 10
    checkpoint = False
    gan = DCGAN(filter_size, num_channel)
    gan.netG.apply(weights_init)
    gan.netD.apply(weights_init)
    train(gan, num_channel, batch_size, learning_rate, L1_lambda, num_epoch, checkpoint)

