import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from CNN_with_deconv import CNNetWithDeconv

STATE_DICTIONARY_PATH = './CIFAR_MODEL_STATE_DICT_TASK_2.pth'
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
BATCH_SIZE = 4


transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def run():
    train_loader, test_loader = get_loaders()

    print('Get CNN model')
    net = CNNetWithDeconv()
    get_model(net, train_loader, train=False)

    print('Testing the model:')
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    show_examples(images, labels, 'Original')
    with torch.no_grad():
        outputs, x_reconstructed = net(images)
    show_examples(x_reconstructed, labels, 'Reconstructed')
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{CLASSES[predicted[j]]:5s}' for j in range(4)))

    calc_accuracy(net, test_loader)


def get_loaders():
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # shuffling to view different tests
    return train_loader, test_loader


def show_examples(images, labels, source):
    title = source + (' images of ' + ', '.join(f'{CLASSES[labels[j]]:5s}' for j in range(BATCH_SIZE)))
    imshow(torchvision.utils.make_grid(images), title)


def imshow(images, title):
    images = images / 2 + 0.5  # un-normalize
    npimg = images.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.savefig(f'./Assignment4/task2_output/{title}')


def get_model(net, train_loader, train=False):
    if not train and os.path.exists(STATE_DICTIONARY_PATH):
        net.load_state_dict(torch.load(STATE_DICTIONARY_PATH))
    else:
        ce_criterion = torch.nn.CrossEntropyLoss()
        mse_criterion = torch.nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        lamb = 2

        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, x_reconstructed = net(inputs)
                loss = ce_criterion(outputs, labels) + (lamb * mse_criterion(inputs, x_reconstructed))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        torch.save(net.state_dict(), STATE_DICTIONARY_PATH)


def calc_accuracy(net, loader):
    correct = 0
    total = 0

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in CLASSES}
    total_pred = {classname: 0 for classname in CLASSES}

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs, _ = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # collect the correct predictions for each class

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1

    print(f'Accuracy of the network on the 10000 test images: {(100 * correct / total):.1f} %')
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == "__main__":
    run()
