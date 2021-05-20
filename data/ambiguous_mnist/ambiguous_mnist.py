import torch
from data.ambiguous_mnist.ambiguous_mnist_dataset import AmbiguousMNIST


def get_loaders(root, train, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataset
    dataset = AmbiguousMNIST(root=root, train=train, device=device,)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)

    return loader
