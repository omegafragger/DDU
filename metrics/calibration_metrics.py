"""
Metrics to measure calibration of a trained deep neural network.
References:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.
"""

import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 20})


# Some keys used for the following dictionaries
COUNT = "count"
CONF = "conf"
ACC = "acc"
BIN_ACC = "bin_acc"
BIN_CONF = "bin_conf"


def _bin_initializer(num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0

    return bin_dict


def _populate_bins(confs, preds, labels, num_bins=10):

    bin_dict = _bin_initializer(num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if bin_dict[binn][COUNT] == 0:
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(bin_dict[binn][COUNT])
    return bin_dict


def expected_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
    return ece


# Calibration error scores in the form of loss metrics
class ECELoss(nn.Module):
    """
    Compute ECE (Expected Calibration Error)
    """

    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


# Methods for plotting reliability diagrams and bin-strength plots
def reliability_plot(confs, preds, labels, num_bins=15, model_name='model'):
    """
    Method to draw a reliability plot from a model's predictions and confidences.
    """
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, bns, align="edge", width=0.03, color="pink", label="Expected")
    plt.bar(bns, y, align="edge", width=0.03, color="blue", alpha=0.5, label="Actual")
    plt.ylabel("Accuracy", fontsize=30)
    plt.xlabel("Confidence", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30, loc='upper left')
    plt.savefig(f'./reliability_plot_{model_name}.pdf')
    plt.savefig(f'./reliability_plot_{model_name}.png')
    plt.show()