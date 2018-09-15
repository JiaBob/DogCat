import torch
import numpy as np

from torch.utils.data import DataLoader, sampler
from torchvision import transforms
import torch.nn.functional as F
from torch import nn, optim

import torchvision

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = DogCat('D:\Dog-data\dog-training\*.tif', transform=transform)  # 20000
test_data = DogCat('D:\Dog-data\dog-test\*.tif', transform=transform)  # 4000

train_sampler = sampler.SubsetRandomSampler(range(18000))
val_sampler = sampler.SubsetRandomSampler(range(18000, 20000))

train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=100)
val_loader = DataLoader(train_data, sampler=val_sampler, batch_size=2000)
test_loader = DataLoader(test_data, batch_size=4000)

resnet = torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

features = resnet.fc.in_features  # 512
resnet.fc = nn.Linear(features, 1)