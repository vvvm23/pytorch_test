import torch
print("Starting Torch version:", torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import torchvision
print("Starting Torchvision version:", torchvision.__version__)
#import torchvision.datasets
from torchvision.datasets import MNIST

# Get datasets if they do not exist
train_set = MNIST(root= "", download=True)
test_set = MNIST(root="", train=False, download=True)

# Print statistics about datasets
print("MNIST loaded. Train:", len(train_set),". Test:", len(test_set))

# CNN class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # CNN ARCH :: [CONV -> POOL] -> [CONV -> POOL] -> FLAT -> DENSE -> DENSE

        # BLOCK 1
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        # BLOCK 2
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        # DENSE BLOCK
        self.fc3 = nn.Linear(1024, 128)
        self.fc3_bn = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        # Forward Block 1
        x = F.relu(self.conv1_bn(self.conv1(F.dropout2d(x, 0.4))))
        x = self.pool1(x)

        # Forward Block 2
        x = F.relu(self.conv2_bn(self.conv2(F.dropout2d(x, 0.4))))
        x = self.pool2(x)

        # Flatten
        x = x.view(-1, self.get_features_nb(x))

        # Dense Block 3
        x = F.relu(self.fc3_bn(self.fc3(F.dropout2d(x, 0.4))))
        x = self.fc4(x)
        return x

    # Function that gets number of features
    def get_features_nb(self, x):
        size = x.size()[1:]
        nb_features = 1
        for s in size:
            nb_features *= s

        #print(nb_features)
        return nb_features

cnn = CNN()

input, t = torch.load("processed/training.pt") # Load training data

target = torch.zeros(60000, 10)
for _ in range(60000):
    target[_, t[_]] = 1 # Explode labels into tensors

torch.stack([input], dim=1, out=input) # Explode 2D image to 3D for conv layers

# Convert to FloatTensor for use with conv layers
# May be pointless to change type here if also changed at input_batch etc.
input = input.type('torch.FloatTensor')
target = target.type('torch.FloatTensor')

print("Defining critic.")
critic = nn.MSELoss() # Mean Square Error Loss
print("Done")
print("Defining Optimiser")
optimiser = optim.Adam(cnn.parameters(), lr=0.001) # Adam Optimiser
print("Done")

bs=20 # Batch Size
epoch=10 # Nb. Epochs
print("Begin training")

with tqdm(total=epoch*60000/bs) as progress_bar: # Create progress bar object
    for _ in (range(epoch)): # Iterate across number of epochs
        for b in (range(int(60000/bs))): # Iterate across number of batches
            progress_bar.update(1) # Increment progress bar

            # Get batch and convert to FloatTensor
            input_batch = input[b*bs:(b+1)*bs].type('torch.FloatTensor')
            target_batch = target[b*bs:(b+1)*bs].type('torch.FloatTensor')

            optimiser.zero_grad() # Zero gradients
            output = cnn(input_batch) # Get output from network
            loss = critic(output, target_batch) # Calculate Loss
            loss.backward() # Backpropogate errors
            optimiser.step() # Apply changes to parameters

print("Training Complete.")
print("Target: ", target[0:5])
print("Actual: ", cnn(input[0:5]))
torch.save(cnn, "saved_model.pth")
