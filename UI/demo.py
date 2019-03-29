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
def load_model(num_epoch=499):
    cloud_computing = True
    filter_size = 64
    num_channel = 3
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    gan = DCGAN(device, filter_size, num_channel, ngpu, cloud_computing)

    batch_size = 64
    learning_rate = 1e-4
    model_path = get_model_name(gan.name, batch_size, learning_rate, num_epoch)
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    gan.load_state_dict(state)
    return gan

# ============================== Generate Image =================================
def generate_image(model, num_channel):

    # load test image
    test_dir = 'test_img'

    if num_channel == 1:
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # test image
    test_edge = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_edge, batch_size=1, num_workers=1)

    test_data = next(iter(test_loader))[0]
    output_img = model.netG(test_data)

    if num_channel == 1:
        output_img = output_img.detach().cpu().numpy().squeeze()
    else:
        output_img = np.transpose(output_img.detach().numpy().squeeze(), [1, 2, 0])

    output_img = (output_img / 2 + 0.5)
    plt.imsave("./output_image.jpg", output_img)
    input_img = cv.imread('./test_img/test_img/test_image.jpg') / 255
    comparison = np.hstack((input_img, output_img))

    # cv.imwrite("./output_image.jpg", output_img)
    plt.imsave("./comparison.jpg", comparison)
    plt.imshow(comparison)

    plt.show()

# =================================== Main ======================================
if __name__ == '__main__':
    model = load_model(359)
    generate_image(model, num_channel=3)

