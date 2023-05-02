import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

def plot_train_test_error(train_errors, test_errors):
	plt.plot([i+1 for i in range(num_epochs)], train_errors, label='Train Error')
	plt.plot([i+1 for i in range(num_epochs)], test_errors, label='Test Error')
	plt.legend()
	plt.xlabel('Epochs')
	plt.ylabel('Error')
	plt.title('Train and Test Error Across Epochs')
	plt.show()

def calc_error(data_loader,model):
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in data_loader:
			images = images.reshape(-1, 28*28).to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		
		return 1 - (correct / total)

def show_misclassified(data_loader,model):
	mistakes = []
	with torch.no_grad():
		for images, labels in data_loader:
			images = images.reshape(-1, 28*28).to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			# correct += (predicted == labels).sum().item()
			for i in range(len(labels)):
				if predicted[i] != labels[i]:
					image = images[i].reshape(28, 28)
					mistakes.append((image, labels[i], predicted[i]))

	fig, axs = plt.subplots(2, 5)
	plt.suptitle("Label | Prediction")
	for ii in range(2):
		for jj in range(5):
			idx = 5 * ii + jj
			axs[ii, jj].imshow(mistakes[idx][0].squeeze())
			axs[ii, jj].set_title(f"{mistakes[idx][1].item()} | {mistakes[idx][2]}")
			axs[ii, jj].axis('off')
	
	plt.show()

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

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        train_errors.append(calc_error(train_loader,model))
        test_errors.append(calc_error(test_loader, model))

    return model, train_errors, test_errors
    
def run():
    final_test_errors = []
    for i in range(1,6):
        _, test_errors, _ = train_model(i)
        final_test_errors.append(test_errors[-1])
        plt.plot(test_errors, label=f"seed {i}")

    plt.title("Test Errors during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    final_test_errors_tensor = torch.tensor(final_test_errors)

    mean_final_test_error = final_test_errors_tensor.mean().item()
    stddev_final_test_error = final_test_errors_tensor.std().item()

    print(f'Mean final Error is {mean_final_test_error}')
    print(f'Standard deviation of final Errors is {stddev_final_test_error}')


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
