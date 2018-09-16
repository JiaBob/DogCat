import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim

from tensorboardX import SummaryWriter
import shutil, os


log_path = 'log'
if log_path not in os.listdir('./'):  # the output of os.listdir does not contain './'
    os.mkdir('./'+log_path)
else:
    shutil.rmtree('./' + log_path, ignore_errors=True)
    os.mkdir('./' + log_path)

model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

writer = SummaryWriter(log_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, criterion, optimizer, train_size, epochs=10):
    losses, val_losses = [], []
    acces, val_acces = [], []

    for epoch in range(epochs):
        loss_sum = 0
        correct = 0
        for img, label in train_loader:
            img = img.to(device)
            label = label.float().to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = model(img).squeeze()
                loss = criterion(out, label).to(device)
                loss.backward()
                optimizer.step()
            loss_sum += loss
            correct += ((label - F.sigmoid(out)).abs() <= 0.5).sum().item()

        val_loss, val_acc = predict(val_loader, model, criterion)
        epoch_loss, epoch_acc = loss_sum / len(train_loader), correct / train_size

        writer.add_scalars('loss', {'train': epoch_loss, 'validation': val_loss}, epoch)
        writer.add_scalars('acc', {'train': epoch_acc, 'validation': val_acc}, epoch)

        val_losses.append(val_loss)
        val_acces.append(val_acc)
        losses.append(epoch_loss)
        acces.append(epoch_acc)

        if (epoch + 1) % 2 == 0:
            print("Epoch {} finished, current loss is {}, \
                  validation accuracy is {}".format(epoch + 1, epoch_loss, val_acc))
            torch.save(model, './{}/{}epoch_result'.format(model_dir, epoch + 1))

    return losses, acces, val_losses, val_acces


def predict(dataset, model, criterion):
    single_batch = next(iter(dataset))
    img, label = single_batch[0], single_batch[1].float()
    with torch.no_grad():
        out = model(img).squeeze()
        loss = criterion(out, label)
        acc = ((label - F.sigmoid(out)).abs() <= 0.5).sum().item()

    return loss.item(), acc / img.size(0)