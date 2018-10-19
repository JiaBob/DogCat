import torch
import numpy as np

from torch.utils.data import DataLoader, sampler
from torchvision import transforms
import torch.nn.functional as F
from torch import nn, optim
import torchvision

from ulti import *
from model import *
from dataloader import *

import argparse
parser = argparse.ArgumentParser(description='pytorch Dog VS Cat')
parser.add_argument('--mode', type=str, default='normal',  choices=['normal', 'transfer'], metavar='M',
                    help='use transfer learning or not (default: normal)')
parser.add_argument('--data', type=str, default='', metavar='D',
                    help='specify the data folder that contains both training set and test set')

args = parser.parse_args()
mode = args.mode
folder_path = args.data

total_size = 20000
val_size = 2000
train_size = total_size - val_size
test_size = 4000

train_sampler = sampler.SubsetRandomSampler(range(train_size))
val_sampler = sampler.SubsetRandomSampler(range(val_size))

train_path = folder_path + '/dog-training/*.tif'
test_path = folder_path + '/dog-test/*.tif'


def transfer():
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = DogCat(train_path, transform=transform)  # 20000
    test_data = DogCat(test_path, transform=transform)  # 4000

    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=100)
    val_loader = DataLoader(train_data, sampler=val_sampler, batch_size=200)  # use single batch may out of memory
    test_loader = DataLoader(test_data, batch_size=200)  # use single batch may out of memory

    resnet = torchvision.models.resnet18(pretrained=True)

    for param in resnet.parameters():
        param.requires_grad = False

    features = resnet.fc.in_features  # 512
    resnet.fc = nn.Linear(features, 1)

    #model = torch.load('9epoch_result')
    model = nn.DataParallel(resnet).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), 1e-3, (0.9, 0.999))

    loss, acc, val_losses, val_acces = train(model, train_loader, val_loader, criterion, optimizer, train_size, val_size)

    _, pred_acc = predict(test_loader, model, criterion, test_size)
    print(pred_acc)


def normal():
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.Resize((62, 62)),
                                    transforms.RandomCrop((32, 32), padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = DogCat(train_path, transform=transform)  # 20000
    test_data = DogCat(test_path, transform=transform)  # 4000

    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=100)
    val_loader = DataLoader(train_data, sampler=val_sampler, batch_size=val_size)  # use single batch
    test_loader = DataLoader(test_data, batch_size=4000)

    model = model = nn.DataParallel(ConvNet()).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), 1e-3, (0.9, 0.999))

    loss, acc, val_losses, val_acces = train(model, train_loader, val_loader, criterion, optimizer, train_size, val_size, epochs=50)
    _, pred_acc = predict(test_loader, model, criterion, test_size)
    print('The test set accuracy is: {}'.format(pred_acc))


if __name__ == '__main__':
    if args.mode == 'normal':
        normal()
    if args.mode == 'transfer':
        transfer()
