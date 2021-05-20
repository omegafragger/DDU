"""
This module contains methods for training models.
"""

import torch
from torch.nn import functional as F
from torch import nn


loss_function_dict = {"cross_entropy": F.cross_entropy}


def train_single_epoch(
    epoch, model, train_loader, optimizer, device, loss_function="cross_entropy", loss_mean=False,
):
    """
    Util method for training a model for a single epoch.
    """
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = loss_function_dict[loss_function](logits, labels)

        if loss_mean:
            loss = loss / len(data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader) * len(data),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / num_samples))
    return train_loss / num_samples


def test_single_epoch(epoch, model, test_val_loader, device, loss_function="cross_entropy"):
    """
    Util method for testing a model for a single epoch.
    """
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for data, labels in test_val_loader:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += loss_function_dict[loss_function](logits, labels).item()
            num_samples += len(data)

    print("======> Test set loss: {:.4f}".format(loss / num_samples))
    return loss / num_samples


def model_save_name(model_name, sn, mod, coeff, seed):
    if sn:
        if mod:
            strn = "_sn_" + str(coeff) + "_mod_"
        else:
            strn = "_sn_" + str(coeff) + "_"
    else:
        if mod:
            strn = "_mod_"
        else:
            strn = "_"

    return str(model_name) + strn + str(seed)
