import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

TRAIN_DATASET = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
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


def task5():
    # Hyper-parameters
    hidden_size = 500
    batch_size = 1000
    learning_rate = 0.01

    torch.manual_seed(1)
    train_loader = DataLoader(dataset=TRAIN_DATASET, shuffle=True, batch_size=batch_size)
    model = train_model(train_loader, hidden_size, learning_rate)
    plot_tsne(model, train_loader)


def train_model(train_loader, hidden_size, learning_rate):
    model = NeuralNet(INPUT_SIZE, hidden_size, NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, INPUT_SIZE).to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def plot_tsne(model, data_loader):
    # Get embeddings for all images in the train set
    embeddings_fc1 = []
    embeddings_input = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(DEVICE)
            out = model.fc1(images.view(images.size(0), -1))
            out = model.relu(out)
            embeddings_fc1.append(out.cpu())
            embeddings_input.append(images.view(images.size(0), -1).cpu())
            labels.append(batch_labels)

        embeddings_fc1 = torch.cat(embeddings_fc1, dim=0)
        embeddings_input = torch.cat(embeddings_input, dim=0)
        labels = torch.cat(labels, dim=0)

    # Use TSNE to reduce the dimensionality of the embeddings to 2D
    tsne = TSNE()
    embeddings_fc1_2d = tsne.fit_transform(embeddings_fc1)
    embeddings_input_2d = tsne.fit_transform(embeddings_input)

    # Plot the 2D embeddings, colored by their corresponding labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.scatter(embeddings_input_2d[:, 0], embeddings_input_2d[:, 1], c=labels, cmap='tab10')
    ax1.set_title('Input Images')
    ax1.axis('off')
    ax1.legend()
    ax2.scatter(embeddings_fc1_2d[:, 0], embeddings_fc1_2d[:, 1], c=labels, cmap='tab10')
    ax2.set_title('Embeddings after first FC layer')
    ax2.axis('off')
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    task5()

