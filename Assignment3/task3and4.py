import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader


TRAIN_DATASET = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
TEST_DATASET = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
NUM_EPOCHS = 5


# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, _input_size, hidden_size, _num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(_input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, _num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def task3():
    # Hyper-parameters
    hidden_size = 500
    batch_size = 100
    learning_rate = 0.001

    validation_error, model_num, epoch_num, correspond_test_error = get_min_validation_error_values(batch_size, hidden_size, learning_rate)
    print('\n\n')
    print(f'Min validation error found in model {model_num}, epoch {epoch_num}.')
    print(f'Min validation error: {validation_error}.')
    print(f'Corresponding test error: {correspond_test_error}.')


def task4():
    # Hyper-parameters
    hidden_sizes = [100, 500]
    batch_sizes = [10, 100, 1000]
    learning_rates = [0.01, 0.001, 0.0001]

    grid_search = []
    for hidden_size in hidden_sizes:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                print(f'Starting hidden_size: {hidden_size}, batch_size: {batch_size}, learning_rate: {learning_rate}')
                _, _, _, test_error = get_min_validation_error_values(batch_size, hidden_size, learning_rate)
                grid_search.append("" + str(hidden_size) + "," + str(batch_size) + "," + str(learning_rate) + "," + str(test_error))
    print(f'{grid_search}')


def get_min_validation_error_values(batch_size, hidden_size, learning_rate):
    np.random.seed(1)
    train_split, validation_split = get_train_validation_split()
    train_loader = DataLoader(dataset=train_split, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_split, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=TEST_DATASET, batch_size=batch_size, shuffle=True)
    validation_errors = []
    test_errors = []
    for i in range(1, 6):
        torch.manual_seed(i)
        print(f'\tStart model {i}')
        _, model_validation_errors, model_test_errors = train_model(train_loader, test_loader, validation_loader,
                                                                    hidden_size, learning_rate)
        validation_errors.extend(model_validation_errors)
        test_errors.extend(model_test_errors)

    min_validation_error = min(validation_errors)
    min_validation_error_index = validation_errors.index(min_validation_error)
    min_validation_error_model = math.floor(min_validation_error_index / 5) + 1
    min_validation_error_epoch = (min_validation_error_index % 5) + 1
    correspond_test_error = test_errors[min_validation_error_index]
    return min_validation_error, min_validation_error_model, min_validation_error_epoch, correspond_test_error


def get_train_validation_split():
    # generate indices: instead of the actual data we pass in integers instead
    train_indices, validation_indices, _, _ = train_test_split(
        range(len(TRAIN_DATASET)), TRAIN_DATASET.targets, stratify=TRAIN_DATASET.targets, test_size=0.1667,
    )
    # generate subset based on indices
    train_split = Subset(TRAIN_DATASET, train_indices)
    validation_split = Subset(TRAIN_DATASET, validation_indices)
    return train_split, validation_split


def calc_error(data_loader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.reshape(-1, 28 * 28).to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return round(1 - (correct / total), 4)


def train_model(train_loader, test_loader, validation_loader, hidden_size, learning_rate):
    model = NeuralNet(INPUT_SIZE, hidden_size, NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    validation_errors = []
    test_errors = []
    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, INPUT_SIZE).to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_error = calc_error(validation_loader, model)
        validation_errors.append(validation_error)
        test_error = calc_error(test_loader, model)
        test_errors.append(test_error)
        print(f'\t\tEpoch {epoch+1} validation error: {validation_error}, test error: {test_error}')

    return model, validation_errors, test_errors


if __name__ == "__main__":
    task3()

