import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
