import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torchvision
import torchvision.transforms as transforms

from CNN_with_deconv import CNNetWithDeconv

STATE_DICTIONARY_PATH = './CIFAR_MODEL_STATE_DICT_TASK_2.pth'
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
BATCH_SIZE = 1


transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def run():
    train_loader, test_loader = get_loaders()

    print('Get CNN model')
    net = CNNetWithDeconv()
    get_model(net)

    train_image, train_image_label = next(iter(train_loader))
    test_image, test_image_label = next(iter(test_loader))
    with torch.no_grad():
        train_output, train_reconstructed = net(train_image)
        test_output, test_reconstructed = net(test_image)
    imshow(torchvision.utils.make_grid(train_reconstructed), "Train image reconstructed")
    imshow(torchvision.utils.make_grid(test_reconstructed), "Test image reconstructed")
        
    show_latent_features(net, train_image, f'Train image - {CLASSES[train_image_label]}')
    show_latent_features(net, test_image, f'Test image - {CLASSES[test_image_label]}')


def get_loaders():
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return train_loader, test_loader


def imshow(images, title):
    images = images / 2 + 0.5  # un-normalize
    npimg = images.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.savefig(f'./Assignment4/task3_output/{title}')


def get_model(net):
    net.load_state_dict(torch.load(STATE_DICTIONARY_PATH))


def show_latent_features(net, image, image_title):
    imshow(image[0], image_title)

    with torch.no_grad():
        x, pool_1_indices = net.get_x_after_first_conv(image)

        for i in range(6):
            x_with_one_channel_only = x.clone()
            for j in range(6):
                if i != j:
                    x_with_one_channel_only[0, j] = 0
            x_reconstructed = net.get_x_reconstructed_after_first_conv(x_with_one_channel_only, pool_1_indices)
            title = image_title + f' - Conv 1 - Channel {i}'
            imshow(x_reconstructed[0], title)

        x, pool_2_indices = net.get_x_after_second_conv(x)
        random.seed(0) # want to see the same channels in both images 
        for i in sorted(random.sample(range(16), k=3)):
            x_with_one_channel_only = x.clone()
            for j in range(16):
                if i != j:
                    x_with_one_channel_only[0, j] = 0
            x_reconstructed = net.get_x_reconstructed_after_second_conv(x_with_one_channel_only, pool_2_indices)
            x_reconstructed = net.get_x_reconstructed_after_first_conv(x_reconstructed, pool_1_indices)
            title = image_title + f' - Conv 2 - Channel {i}'
            imshow(x_reconstructed[0], title)


if __name__ == "__main__":
    run()
