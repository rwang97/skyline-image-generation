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

# ============================ Class for two dataset ================================
# https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/2
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

# =================================== Load Data ======================================
def get_data_loader(num_channel, batch_size):
    # We transform them to Tensors of normalized range [-1, 1].
    if num_channel == 1:
        real_dir = './data'
        input_dir = './denoise'
        test_dir = './test'
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        real_dir = './data'
        input_dir = './denoise'
        test_dir = './test'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    input_set = torchvision.datasets.ImageFolder(root=input_dir, transform=transform)  # input images for generator
    real_set = torchvision.datasets.ImageFolder(root=real_dir, transform=transform) # real images

    np.random.seed(1000)  # Fixed numpy random seed for reproducible shuffling
    train_loader = torch.utils.data.DataLoader(ConcatDataset(input_set, real_set), batch_size=batch_size, shuffle=True, num_workers=1)

    # test image
    test_edge = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_edge, batch_size=1, num_workers=1)

    return train_loader, test_loader

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

def test_output(model, test_loader, num_channel, epoch, checkpoint=False):
    test_data = next(iter(test_loader))[0]
    test_fake = model.netG(test_data.to(device) if cloud_computing else test_data)
    if num_channel == 1:
        test_fake = test_fake.detach().cpu().numpy().squeeze()
    else:
        test_fake = np.transpose(test_fake.detach().cpu().numpy().squeeze(), [1, 2, 0])

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
def train(model, device, num_channel=1, batch_size=32, learning_rate=1e-4, L1_lambda=10, num_epochs=5, checkpoint=False, cloud_computing=False):

    if checkpoint:
        if os.path.exists('./checkpoints'):
            shutil.rmtree('./checkpoints')
        os.makedirs('./checkpoints')

    # clean the fake directory
    if os.path.exists('./data/Fake'):
        shutil.rmtree('./data/Fake')
    os.makedirs('./data/Fake')
    
    # load training data
    train_loader, test_loader = get_data_loader(num_channel, batch_size)

    # loss function and optimizer
    BCE_Loss = nn.BCELoss()
    L1_Loss = nn.L1Loss()
    optimizerD = optim.Adam(model.netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(model.netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    real_label = 1
    fake_label = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        fake_avg = 0
        real_avg = 0
        # output result to disk
        test_output(model, test_loader, num_channel, epoch, checkpoint)
        
        for i, (input, real) in enumerate(train_loader, 0):
            input_data = input[0].to(device) if cloud_computing else input[0]
            real_data = real[0].to(device) if cloud_computing else real[0]
            ############################
            # (1) Update D network
            ############################
            model.netD.zero_grad()
            # Train with all-fake batch
            # Generate fake image batch with G
            fake = model.netG(input_data).to(device) if cloud_computing else model.netG(input_data)

            # Forward pass real batch through D
            output = model.netD(input_data, real_data)
            label = torch.full(output.shape, real_label)
            if cloud_computing == True:
                label = label.to(device)
            # Calculate loss on all-real batch
            loss_D_real = BCE_Loss(output, label)
            D_real = output.mean().item()
            real_avg += D_real

            # Classify all fake batch with D
            output = model.netD(input_data, fake.detach())
            label = torch.full(output.shape, fake_label)
            if cloud_computing == True:
                label = label.to(device)
            # Calculate D's loss on the all-fake batch
            loss_D_fake = BCE_Loss(output, label)
            D_fake = output.mean().item()
            fake_avg += D_fake

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
            output = model.netD(input_data, fake)
            # Calculate G's loss based on this output
            loss_G_BCE = BCE_Loss(output, label)
            loss_G_L1 = L1_Loss(fake, real_data) * L1_lambda
            loss_G = loss_G_BCE + loss_G_L1
            # Calculate gradients for G
            loss_G.backward()
            optimizerG.step()

            # Output training stats
            print("epoch: " + str(epoch) + ", iteration: " + str(i))
            print("d_loss is: " + str(float(loss_D)) + ", g_loss is: " + str(float(loss_G)))
            print("d_real is: " + str(D_real) + ", d_fake is: " + str(D_fake))
            print("================================================================")

            torch.cuda.empty_cache()

        # Calculate fake_avg and real_avg
        fake_avg = fake_avg/(i+1)
        real_avg = real_avg/(i+1)

        if epoch % 5 == 4:
            # Train D network separately until
            print("======================== starting to train discriminator =======================")
            while (real_avg < 0.5 and fake_avg > 0.4):
                fake_avg = 0
                real_avg = 0
                for j, (input, real) in enumerate(train_loader, 0):
                    input_data = input[0].to(device) if cloud_computing else input[0]
                    real_data = real[0].to(device) if cloud_computing else real[0]
                    model.netD.zero_grad()
                    # Train with all-fake batch
                    # Generate fake image batch with G
                    fake = model.netG(input_data).to(device) if cloud_computing else model.netG(input_data)

                    # Forward pass real batch through D
                    output = model.netD(input_data, real_data)
                    label = torch.full(output.shape, real_label)
                    if cloud_computing == True:
                        label = label.to(device)
                    # Calculate loss on all-real batch
                    loss_D_real = BCE_Loss(output, label)
                    D_real = output.mean().item()
                    real_avg += D_real

                    # Classify all fake batch with D
                    output = model.netD(input_data, fake.detach())
                    label = torch.full(output.shape, fake_label)
                    if cloud_computing == True:
                        label = label.to(device)
                    # Calculate D's loss on the all-fake batch
                    loss_D_fake = BCE_Loss(output, label)
                    D_fake = output.mean().item()
                    fake_avg += D_fake

                    # Add the gradients from the all-real and all-fake batches
                    loss_D = (loss_D_real + loss_D_fake) / 2
                    loss_D.backward()
                    optimizerD.step()

                    # Output training stats
                    
                    print("epoch: " + str(epoch) + ", iteration: " + str(j))
                    print("d_loss is: " + str(float(loss_D)) + ", d_real is: " + str(D_real) + ", d_fake is: " + str(D_fake))
                    print("================================================================")

                    torch.cuda.empty_cache()
                fake_avg = fake_avg / (j + 1)
                real_avg = real_avg / (j + 1)

        if checkpoint and epoch % 10 == 9:
            # Save the current model (checkpoint) to a file
            model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
            torch.save(model.state_dict(), model_path)
    # output final result to disk
    test_output(model, test_loader, num_channel, epoch)


# =================================== Main ======================================
if __name__ == '__main__':
    filter_size = 64
    num_channel = 1
    num_epoch = 200
    batch_size = 64
    learning_rate = 2e-4
    L1_lambda = 100
    checkpoint = True
    ngpu = 1
    cloud_computing = True
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    gan = DCGAN(device, filter_size, num_channel, ngpu, cloud_computing)
    gan.netG.apply(weights_init)
    gan.netD.apply(weights_init)
    train(gan, device, num_channel, batch_size, learning_rate, L1_lambda, num_epoch, checkpoint, cloud_computing)

