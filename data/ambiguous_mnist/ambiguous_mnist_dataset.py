import os
import torch
import random
import numpy as np
from torch.utils.data.dataset import Dataset

"""
Loads the ambiguous MNIST dataset. 
Every image in the dataset is 28x28 pixels.
Set root to point to the dataset folder.
"""


class AmbiguousMNIST(Dataset):
    def __init__(self, root, train=True, device=None):
        # Scale data to [0,1]
        self.data = torch.load(os.path.join(root, "amnist_samples.pt")).to(device)
        self.targets = torch.load(os.path.join(root, "amnist_labels.pt")).to(device)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        num_multi_labels = self.targets.shape[1]

        self.data = self.data.expand(-1, num_multi_labels, 28, 28).reshape(-1, 1, 28, 28)
        self.targets = self.targets.reshape(-1)

        data_range = slice(None, 60000) if train else slice(60000, None)
        self.data = self.data[data_range]
        self.targets = self.targets[data_range]

    # The number of items in the dataset
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
