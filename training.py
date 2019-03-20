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
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # load each dataset with corresponding folders
    realset = torchvision.datasets.ImageFolder(root='./data', transform=transform)
    real_loader = torch.utils.data.DataLoader(realset, batch_size=batch_size,
                                               num_workers=1)

    # load each dataset with corresponding folders
    edgeset = torchvision.datasets.ImageFolder(root='./generator_input', transform=transform)
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

# save model weights
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

# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def train(model, batch_size=32, learning_rate=1e-4, L1_lambda=10, num_epochs=5):
    # clean the fake directory
    if os.path.exists('./data/Fake'):
        shutil.rmtree('./data/Fake')
    os.makedirs('./data/Fake')

    if os.path.exists('./checkpoints'):
        shutil.rmtree('./checkpoints')
    os.makedirs('./checkpoints')
    
    # load training data
    real_loader, edge_loader = get_data_loader(batch_size)

    # loss function and optimizer
    BCE_Loss = nn.BCELoss()
    # MSE_Loss = nn.MSELoss()
    L1_Loss = nn.L1Loss()
    optimizerD = optim.Adam(model.netD.parameters(), lr=learning_rate)
    optimizerG = optim.Adam(model.netG.parameters(), lr=learning_rate)

    real_label = 1
    fake_label = 0

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_edge = torchvision.datasets.ImageFolder(root='./test', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_edge, batch_size=1,num_workers=1)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        edge_iter = iter(edge_loader)

        test_fake = model.netG(next(iter(test_loader))[0])
        test_fake = np.transpose(test_fake.detach().numpy().squeeze(), [1, 2, 0])
        test_fake = (test_fake / 2 + 0.5) * 255
        cv.imwrite("./data/Fake/" + str(epoch) + ".jpg", test_fake)

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

            # Classify all fake batch with D
            output = model.netD(edge_batch[0], fake.detach())
            label = torch.full(output.shape, fake_label)
            # Calculate D's loss on the all-fake batch
            loss_D_fake = BCE_Loss(output, label)

            # Add the gradients from the all-real and all-fake batches
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            model.netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = model.netD(edge_batch[0], fake.detach())
            # Calculate G's loss based on this output
            loss_G_BCE = BCE_Loss(output, label)
            loss_G_L1 = L1_Loss(fake, real_data[0]) * L1_lambda
            loss_G = loss_G_BCE + loss_G_L1
            # Calculate gradients for G
            loss_G.backward()
            # Update G
            optimizerG.step()

            # Output training stats
            print("epoch: " + str(epoch) + ", iteration: " + str(i) + ", d_loss is: " + str(float(loss_D)) + ", g_loss is: " + str(float(loss_G)))
            print("=============================================================")
        
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
        torch.save(model.state_dict(), model_path)

    test_fake_result = model.netG(next(iter(test_loader))[0])
    test_fake_result = np.transpose(test_fake_result.detach().numpy().squeeze(), [1, 2, 0]) * 255
    test_fake_result = (test_fake_result / 2 + 0.5) * 255
    cv.imwrite("./data/Fake/result.jpg", test_fake_result)


###############################################################################
# Main Function
if __name__ == '__main__':
    filter_size = 64
    gan = DCGAN(filter_size)
    gan.netG.apply(weights_init)
    gan.netD.apply(weights_init)
    num_epoch = 10
    batch_size = 32
    learning_rate = 1e-4
    L1_lambda = 10
    train(gan, batch_size, learning_rate, L1_lambda, num_epoch)
    pass

