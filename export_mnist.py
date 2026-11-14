# This script trains a 2-layer neural network(782 -> 128 -> 10) on MNIST for 1 epoch
# transposes the weight matrices to match my c++ (input x weight) layout and exports the 
# weight, biases, and a single test image as raw 64-bit float .bin files.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# 1. Define the exact 2-layer architecture matching C++ forward_pass
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 2. Load MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. Quick training (1 epoch is enough for ~95% accuracy)
model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training MNIST model...")
model.train()
for images, labels in train_loader:
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

# 4. Export weights and biases as raw FP64 (C++ double) binary files
# Note: PyTorch stores weights as [out, in], so we transpose (.T) to [in, out] for our C++ matmul
model.eval()
model.fc1.weight.detach().numpy().T.astype(np.float64).tofile("w1.bin")
model.fc1.bias.detach().numpy().astype(np.float64).tofile("b1.bin")
model.fc2.weight.detach().numpy().T.astype(np.float64).tofile("w2.bin")
model.fc2.bias.detach().numpy().astype(np.float64).tofile("b2.bin")

# 5. Export 1,000 test images and labels for benchmarking
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
images_1k, labels_1k = next(iter(test_loader))

images_1k.view(1000, 784).numpy().astype(np.float64).tofile("test_images_1k.bin")
labels_1k.numpy().astype(np.float64).tofile("test_labels_1k.bin")
images_1k[0].view(1, 784).numpy().astype(np.float64).tofile("test_img.bin")

print("Exported 1,000 test images (test_images_1k.bin) and labels (test_labels_1k.bin)!")