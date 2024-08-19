import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Custom MNIST Dataset class to load data from IDX files
class MNISTDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        self.images = self.load_images(image_file)
        self.labels = self.load_labels(label_file)
        self.transform = transform

    def load_images(self, filename):
        with open(filename, 'rb') as f:
            # Read the magic number, number of images, rows, and columns
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # Read the image data and reshape it into (num_images, rows, cols)
            image_data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
            return image_data

    def load_labels(self, filename):
        with open(filename, 'rb') as f:
            # Read the magic number and number of labels
            magic, num_labels = struct.unpack(">II", f.read(8))
            # Read the label data
            label_data = np.fromfile(f, dtype=np.uint8)
            return label_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the image and label at the given index
        image = self.images[idx]
        label = self.labels[idx]

        # Apply any transformations (e.g., convert image to tensor)
        if self.transform:
            image = self.transform(image)

        return image, label


# Define a transformation to convert images to PyTorch tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST dataset
])

# Load the dataset using the custom loader
train_dataset = MNISTDataset('./data/train-images.idx3-ubyte', './data/train-labels.idx1-ubyte', transform=transform)
test_dataset = MNISTDataset('./data/t10k-images.idx3-ubyte', './data/t10k-labels.idx1-ubyte', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Input layer to hidden layer (28x28 flattened to 784 -> 128 neurons)
        self.fc1 = nn.Linear(28 * 28, 128)
        # Hidden layer to output layer (128 -> 10 neurons for 10 classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten the input (28x28 pixels to 784)
        x = x.view(-1, 28 * 28)
        # Hidden layer with ReLU activation
        x = torch.relu(self.fc1(x))
        # Output layer (raw logits, softmax will be applied later in the loss function)
        x = self.fc2(x)
        return x


# Set up device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the network, define the loss function and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data to device

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)

        # Compute the loss
        loss = criterion(outputs, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluation loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Disable gradients during evaluation
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # Forward pass
        outputs = model(data)

        # Get the predicted class by taking the argmax of the outputs (highest probability)
        _, predicted = torch.max(outputs, 1)

        # Update the correct and total counts
        correct += (predicted == target).sum().item()
        total += target.size(0)

# Calculate and print accuracy
accuracy = correct / total * 100
print(f'Accuracy on test set: {accuracy:.2f}%')
