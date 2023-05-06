import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def calc_error(data_loader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return 1 - (correct / total)


def train_model(seed=-1):
    if seed != -1:
        torch.manual_seed(seed)

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    train_errors = []
    test_errors = []
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        train_errors.append(calc_error(train_loader, model))
        test_errors.append(calc_error(test_loader, model))

    return model, train_errors, test_errors


def plot_tsne(model, data_loader):
    # Get embeddings for all images in the train set
    embeddings_fc1 = []
    embeddings_input = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(device)
            out = model.fc1(images.view(images.size(0), -1))
            out = model.relu(out)
            embeddings_fc1.append(out.cpu())
            embeddings_input.append(images.view(images.size(0), -1).cpu())
            labels.append(batch_labels)

        embeddings_fc1 = torch.cat(embeddings_fc1, dim=0)
        embeddings_input = torch.cat(embeddings_input, dim=0)
        labels = torch.cat(labels, dim=0)

    # Use TSNE to reduce the dimensionality of the embeddings to 2D
    tsne = TSNE(n_components=2, random_state=42)
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


def run():
    model, _, _ = train_model()

    plot_tsne(model, train_loader)


# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
if __name__ == "__main__":
    run()

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
