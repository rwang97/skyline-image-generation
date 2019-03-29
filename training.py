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
    real_dir = './data'
    input_dir = './mor_edges'
    test_dir = './test'
    transform_gray = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform_color = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    input_set = torchvision.datasets.ImageFolder(root=input_dir, transform=transform_gray)  # input images for generator
    real_set = torchvision.datasets.ImageFolder(root=real_dir, transform=transform_color) # real images

    np.random.seed(1000)  # Fixed numpy random seed for reproducible shuffling
    train_loader = torch.utils.data.DataLoader(ConcatDataset(input_set, real_set), batch_size=batch_size, shuffle=True, num_workers=1)

    # test image
    test_edge = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_gray)
    test_loader = torch.utils.data.DataLoader(test_edge, batch_size=1, num_workers=1)

    return train_loader, test_loader

# ============================= Weight Initialization ======================================
# Weight initialization
# custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

def test_output(model_R, model_G, model_B, test_loader, num_channel, epoch):
    for i, image in enumerate(test_loader, 0):
        test_data_np = image[0]
        # test_fake = np.transpose(test_fake.detach().cpu().numpy().squeeze(), [1, 2, 0])
        test_fake_R = (model_R.netG(test_data_np.to(device) if cloud_computing else test_data_np)).detach().cpu().numpy().squeeze(0)
        test_fake_G = (model_G.netG(test_data_np.to(device) if cloud_computing else test_data_np)).detach().cpu().numpy().squeeze(0)
        test_fake_B = (model_B.netG(test_data_np.to(device) if cloud_computing else test_data_np)).detach().cpu().numpy().squeeze(0)
        test_fake = np.transpose(np.concatenate((test_fake_R, test_fake_G, test_fake_B),axis=0), [1, 2, 0])

        test_fake = (test_fake / 2 + 0.5)
        plt.imsave("./data/Fake/" + str(i) + '/' + str(epoch) + ".jpg", test_fake)
        torch.cuda.empty_cache()

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
def train(model_R, model_G, model_B, device, num_channel=1, batch_size=32, learning_rate=1e-4, L1_lambda=10, num_epochs=5, checkpoint=False, cloud_computing=False):

    if checkpoint:
        if os.path.exists('./checkpoints'):
            shutil.rmtree('./checkpoints')
        os.makedirs('./checkpoints')

    # clean the fake directory
    if os.path.exists('./data/Fake'):
        shutil.rmtree('./data/Fake')
    os.makedirs('./data/Fake')

    for i, image in enumerate(os.listdir('./test/test')):
        if os.path.exists('./data/Fake/' + str(i)):
            shutil.rmtree('./data/Fake/' + str(i))
        os.makedirs('./data/Fake/' + str(i))
    
    # load training data
    train_loader, test_loader = get_data_loader(3, batch_size)

    # loss function and optimizer
    BCE_Loss = nn.BCELoss()
    L1_Loss = nn.L1Loss()
    optimizerD_R = optim.Adam(model_R.netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG_R = optim.Adam(model_R.netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerD_G = optim.Adam(model_G.netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG_G = optim.Adam(model_G.netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerD_B = optim.Adam(model_B.netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG_B = optim.Adam(model_B.netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    real_label = 1
    fake_label = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):

        # output result to disk
        # if epoch % 5 == 4:
        test_output(model_R, model_G, model_B, test_loader, 3, epoch)

        for i, (input, real) in enumerate(train_loader, 0):
            # input_data_np = input[0].numpy()
            # input_R_tensor = torch.from_numpy(input_data_np[:,0,:,:])
            # input_G_tensor = torch.from_numpy(input_data_np[:,1,:,:])
            # input_B_tensor = torch.from_numpy(input_data_np[:,2,:,:])
            input_tensor = input[0].to(device)
            real_data_np = real[0].numpy()
            real_R_tensor = (torch.from_numpy(real_data_np[:,0,:,:])).unsqueeze(1)
            real_G_tensor = (torch.from_numpy(real_data_np[:,1,:,:])).unsqueeze(1)
            real_B_tensor = (torch.from_numpy(real_data_np[:,2,:,:])).unsqueeze(1)
            if cloud_computing:
                # input_tensor.to(device)
                real_R_tensor.to(device)
                real_G_tensor.to(device)
                real_B_tensor.to(device)

            for color in ['R', 'G', 'B']:
                # Select network, data, etc.
                if color == 'R':
                    model, input_data, real_data, optimizerD, optimizerG = model_R, input_tensor, real_R_tensor, optimizerD_R, optimizerG_R
                elif color == 'G':
                    model, input_data, real_data, optimizerD, optimizerG = model_G, input_tensor, real_G_tensor, optimizerD_G, optimizerG_G
                else:
                    model, input_data, real_data, optimizerD, optimizerG = model_B, input_tensor, real_B_tensor, optimizerD_B, optimizerG_B
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

                # Classify all fake batch with D
                output = model.netD(input_data, fake.detach())
                label = torch.full(output.shape, fake_label)
                if cloud_computing == True:
                    label = label.to(device)
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
                output = model.netD(input_data, fake)
                # Calculate G's loss based on this output
                loss_G_BCE = BCE_Loss(output, label)
                loss_G_L1 = L1_Loss(fake, real_data) * L1_lambda
                loss_G = loss_G_BCE + loss_G_L1
                # Calculate gradients for G
                loss_G.backward()
                optimizerG.step()

                # Output training stats
                print("epoch: " + str(epoch) + ", iteration: " + str(i) + ", color: " + color)
                print("d_loss is: " + str(float(loss_D)) + ", g_loss is: " + str(float(loss_G)))
                print("d_real is: " + str(D_real) + ", d_fake is: " + str(D_fake))
                print("================================================================")

                torch.cuda.empty_cache()

        # if checkpoint and epoch % 10 == 9:
        #     # Save the current model (checkpoint) to a file
        #     model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
        #     torch.save(model.state_dict(), model_path)

    # output final result to disk
    test_output(model_R, model_G, model_B, test_loader, 3, epoch)


# =================================== Main ======================================
if __name__ == '__main__':
    filter_size = 64
    num_channel = 1
    num_epoch = 200
    batch_size = 64
    learning_rate = 1e-4
    L1_lambda = 50
    checkpoint = True
    ngpu = 1
    cloud_computing = True
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    gan_R = DCGAN(device, filter_size, num_channel, 'R', ngpu, cloud_computing)
    gan_G = DCGAN(device, filter_size, num_channel, 'G', ngpu, cloud_computing)
    gan_B = DCGAN(device, filter_size, num_channel, 'B', ngpu, cloud_computing)
    train(gan_R, gan_G, gan_B, device, num_channel, batch_size, learning_rate, L1_lambda, num_epoch, checkpoint, cloud_computing)

