import torch
print("Torch version:", torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from tqdm import tqdm

import torchvision
print("Torchvision version:", torchvision.__version__)
import torchvision.datasets
from torchvision.datasets import MNIST

train_set = MNIST(root= "", download=True)
test_set = MNIST(root="", train=False, download=True)

print("MNIST loaded. Train:", len(train_set),". Test:", len(test_set))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.fc3 = nn.Linear(1024, 128)
        self.fc3_bn = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(F.dropout2d(x, 0.4))))
        x = self.pool1(x)
        x = F.relu(self.conv2_bn(self.conv2(F.dropout2d(x, 0.4))))
        x = self.pool2(x)

        x = x.view(-1, self.get_features_nb(x))
        x = F.relu(self.fc3_bn(self.fc3(F.dropout2d(x, 0.4))))
        x = self.fc4(x)
        return x

    def get_features_nb(self, x):
        size = x.size()[1:]
        nb_features = 1
        for s in size:
            nb_features *= s

        #print(nb_features)
        return nb_features

cnn = CNN()

input, t = torch.load("processed/training.pt")

target = torch.zeros(60000, 10)
for _ in range(60000):
    target[_, t[_]] = 1

torch.stack([input], dim=1, out=input)

input = input.type('torch.FloatTensor')
target = target.type('torch.FloatTensor')

print(input.shape)
print(target.shape)

#target = torch.zeros((2, 10))
#input = torch.zeros((2, 1, 28, 28))
#
#target[0] = torch.FloatTensor(F.softmax(torch.randn(10), dim=-1))
#target[1] = torch.FloatTensor(F.softmax(torch.randn(10), dim=-1))
#
#input[0] = torch.FloatTensor(torch.randn(1, 28, 28))
#input[1] = torch.FloatTensor(torch.randn(1, 28, 28))

print("Defining critic.")
critic = nn.MSELoss()
print("Done")
print("Defining Optimiser")
optimiser = optim.Adam(cnn.parameters(), lr=0.001)
print("Done")

bs=20
epoch=10
print("Begin training")

with tqdm(total=epoch*60000/bs) as progress_bar:
    for _ in (range(epoch)):
        for b in (range(int(60000/bs))):
            #os.system("clear")
            progress_bar.update(1)
            #print("Training CNN:", (_*60000/bs + b)/(60000*epoch/bs) * 100, "%")
            #print("Epoch", _)
            #print("Loading Batch", b)

            input_batch = input[b*bs:(b+1)*bs].type('torch.FloatTensor')
            target_batch = target[b*bs:(b+1)*bs].type('torch.FloatTensor')
            #print(input_batch.shape)
            #print(target_batch.shape)
            #print("Done.")
            optimiser.zero_grad()
            output = cnn(input_batch)
            loss = critic(output, target_batch)
            loss.backward()
            optimiser.step()

print("Target: ", target[0:5])
print("Actual: ", cnn(input[0:5]))