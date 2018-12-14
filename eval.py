from model import CNN

import torch
print("Starting Torch version:", torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import torchvision
print("Starting Torchvision version:", torchvision.__version__)
#import torchvision.datasets
from torchvision.datasets import MNIST

import matplotlib
import matplotlib.pyplot as plt
print("Starting matplotlib version:", matplotlib.__version__)

# Get test set if they do not exist
test_set = MNIST(root="", train=False, download=True)

# Print statistics about datasets
print("MNIST loaded. Test:", len(test_set))

# Load model
print("Loading CNN from file.")
cnn = torch.load("saved_model.pth")
print(cnn)
print("Loaded CNN.")

print("Evaluating network")
cnn.eval()
correct = 0
total = 0

inputs, labels = torch.load("processed/test.pt") # Load training data
torch.stack([inputs], dim=1, out=inputs) # Explode 2D image to 3D for conv layers
inputs = inputs.type('torch.FloatTensor')

with torch.no_grad():
    for _ in tqdm(range(len(inputs))):
        input = torch.stack([inputs[_]], dim=1, out=inputs[_])
        output = cnn.forward(input)
        prediction = torch.argmax(output)
        if prediction == labels[_]:
            correct += 1
        total += 1

print("Evaluation complete. Network accuracy:", correct/total * 100, "%")