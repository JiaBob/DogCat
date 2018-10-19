import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim

from tensorboardX import SummaryWriter
import shutil, os, time

log_folder = './runs'  # for comparing several training result
log_path = 'log2'
log = '{}/{}'.format(log_folder, log_path)

# check if 'log_folder' exist, if not create it, if yes but 'log' already exists
# remove 'log', final create 'log' in 'log_folder'
if not os.path.exists(log_folder):
    print(log_folder)
    os.makedirs(log)
elif log_path in os.listdir(log_folder):  # the output of os.listdir does not contain './'
    if os.path.exists(log):
        shutil.rmtree(log, ignore_errors=True)
    os.makedirs(log)

model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

writer = SummaryWriter(log)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, criterion, optimizer, train_size, val_size, epochs=10, verbose=2, save=False):
    losses, val_losses = [], []
    acces, val_acces = [], []

    for epoch in range(epochs):
        start_time = time.time()
        loss_sum = 0
        correct = 0
        for img, label in train_loader:
            img = img.to(device)
            label = label.float().to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = model(img).squeeze()
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
            loss_sum += loss
            correct += ((label - torch.sigmoid(out)).abs() <= 0.5).sum().item()

        val_loss, val_acc = predict(val_loader, model, criterion, val_size)
        epoch_loss, epoch_acc = loss_sum / len(train_loader), correct / train_size

        writer.add_scalars('loss', {'train': epoch_loss, 'validation': val_loss}, epoch)
        writer.add_scalars('acc', {'train': epoch_acc, 'validation': val_acc}, epoch)

        val_losses.append(val_loss)
        val_acces.append(val_acc)
        losses.append(epoch_loss)
        acces.append(epoch_acc)

        if verbose and (epoch + 1) % verbose == 0:
            print("Epoch {} finished, takes {:.1f}s, current loss is {}, \
                  validation accuracy is {}".format(epoch + 1, time.time() - start_time, epoch_loss, val_acc))
        if save:
            torch.save(model, './{}/{}epoch_result'.format(model_dir, epoch + 1))

    return losses, acces, val_losses, val_acces


def predict(dataset, model, criterion, datasize):
    loss_sum = 0
    correct = 0
    with torch.no_grad():
        for img, label in dataset:
            img, label = img.to(device), label.float().to(device)
            out = model(img).squeeze()
            loss = criterion(out, label)

            loss_sum += loss
            correct += ((label - torch.sigmoid(out)).abs() <= 0.5).sum().item()
    loss, acc = loss_sum / len(dataset), correct / datasize

    return loss.item(), acc