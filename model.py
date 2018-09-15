import torch.nn.functional as F
from torch import nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(  # 3 * 32 * 32 -> 32 * 16 * 16
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(  # 32 * 16 * 16 -> 64 * 8 * 8
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x