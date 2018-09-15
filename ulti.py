import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim


def train(model, train_loader, criterion, optimizer, train_size, epochs=10):
    losses, val_losses = [], []
    acces, val_acces = [], []

    for epoch in range(epochs):
        loss_sum = 0
        correct = 0
        for img, label in train_loader:
            label = label.float()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = model(img).squeeze()
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
            loss_sum += loss

            correct += ((label - F.sigmoid(out)).abs() <= 0.5).sum().item()

        val_loss, val_acc = predict(val_loader, model, criterion)
        epoch_loss, epoch_acc = loss_sum / len(train_loader), correct / train_size

        val_losses.append(val_loss)
        val_acces.append(val_acc)
        losses.append(epoch_loss)
        acces.append(epoch_acc)

        if (epoch + 1) % 2 == 0:
            print("Epoch {} finished, current loss is {}, \
                  validation accuracy is {}".format(epoch + 1, epoch_loss, val_acc))
            torch.save(model, '{}epoch_result'.format(epoch + 1))

    return losses, acces, val_losses, val_acces


def predict(dataset, model, criterion):
    single_batch = next(iter(dataset))
    img, label = single_batch[0], single_batch[1].float()
    with torch.no_grad():
        out = model(img).squeeze()
        loss = criterion(out, label)
        acc = ((label - F.sigmoid(out)).abs() <= 0.5).sum().item()

    return loss.item(), acc / img.size(0)