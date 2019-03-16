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

###############################################################################
# Data Loading
def get_data_loader(batch_size):
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load each dataset with corresponding folders
    trainset = torchvision.datasets.ImageFolder(root='./data', transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=1)

    return train_loader


def train(model, batch_size=64, learning_rate=1e-4):
    # load training data
    train_loader = get_data_loader(batch_size)

    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=learning_rate)
    optimizer_g = optim.Adam(model.generator.parameters(), lr=learning_rate)

    # measure time
    start_time = time.time()

    total_train_loss = 0.0
    # train d
    while(get_accuracy())

    for i, data in enumerate(train_loader, 0):
        # Get the inputs
        inputs, labels = data

        # Forward pass, backward pass, and optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Calculate the statistics
        total_train_loss += loss.item()

###############################################################################
# Main Function
if __name__ == '__main__':
    filter_size = 64
    gan = DCGAN(filter_size)
    train(gan)
    #train_model(gan)