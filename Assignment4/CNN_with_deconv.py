import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNetWithDeconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1T = nn.ConvTranspose2d(6, 3, 5)
        self.conv2T = nn.ConvTranspose2d(16, 6, 5)

    def forward(self, x):
        x, pool_1_indices = self.pool(F.relu(self.conv1(x)))
        x, pool_2_indices = self.pool(F.relu(self.conv2(x)))

        reconstructed_x = self.conv2T(F.relu(self.unpool(x, pool_2_indices)))
        reconstructed_x = self.conv1T(F.relu(self.unpool(reconstructed_x, pool_1_indices)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, reconstructed_x

    def get_x_after_first_conv(self, x):
        return self.pool(F.relu(self.conv1(x)))

    def get_x_reconstructed_after_first_conv(self, x, pool_1_indices):
        return self.conv1T(F.relu(self.unpool(x, pool_1_indices)))

    def get_x_after_second_conv(self, x):
        return self.pool(F.relu(self.conv2(x)))

    def get_x_reconstructed_after_second_conv(self, x, pool_2_indices):
        return self.conv2T(F.relu(self.unpool(x, pool_2_indices)))
