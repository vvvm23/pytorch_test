from model import CNN

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

# Print statistics about datasets
print("MNIST loaded. Train:", len(train_set),". Test:", len(test_set))


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