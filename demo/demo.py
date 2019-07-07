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

# =================================== Checkpoint ======================================
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "./model_{0}_bs{1}_lr{2}_epoch{3}".format(name,batch_size,learning_rate, epoch)
    
    return path

# ================================ Load Model ===================================
def load_model(num_channel=3, num_epoch=899):
    # example parameters we used to train our model
    cloud_computing = True
    filter_size = 64
    ngpu = 1
    batch_size = 64
    learning_rate = 1e-4
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    gan = DCGAN(device, filter_size, num_channel, ngpu, cloud_computing)

    model_path = get_model_name(gan.name, batch_size, learning_rate, num_epoch)
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    gan.load_state_dict(state)
    return gan

# ============================== Generate Image =================================
def generate_image(model=None, num_channel=3):

    if os.path.exists('output'):
        shutil.rmtree('output')

    os.makedirs('output')

    if os.path.exists('comparison'):
        shutil.rmtree('comparison')

    os.makedirs('comparison')

    # load test image under "test" folder
    test_dir = 'test'
    assert(num_channel==3)

    # if num_channel == 1:
    #     transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # else:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # test image
    test_edge = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_edge, batch_size=1, num_workers=1)

    # traversing the test folder to generate a new image for each sketch 
    for i, image in enumerate(test_loader, 0):
        test_data = image[0]
        output_img = model.netG(test_data)
        # if num_channel == 1:
        #     output_img = output_img.detach().cpu().numpy().squeeze()
        # else:
        output_img = np.transpose(output_img.detach().numpy().squeeze(), [1, 2, 0])

        output_img = (output_img / 2 + 0.5)
        plt.imsave("./output/" + str(i) + ".jpg", output_img)

        temp_test_data = np.transpose(test_data.detach().cpu().numpy().squeeze(), [1, 2, 0])
        temp_test_data = (temp_test_data / 2 + 0.5)
        comparison = np.hstack((temp_test_data, output_img))

        plt.imsave("./comparison/" + str(i) + ".jpg", comparison)


# =================================== Main ======================================
if __name__ == '__main__':
    # we only support rgb
    num_channel = 3
    model = load_model(num_channel, 899)
    generate_image(model, num_channel)
    print("Done generating images, please check 'output' and 'comparison' directory")

