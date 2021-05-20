"""
Pytorch implementation of VGG models.
Reference:
[1] . Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.
"""

import torch
import torch.nn as nn

from net.spectral_normalization.spectral_norm_conv_inplace import spectral_norm_conv
from net.spectral_normalization.spectral_norm_fc import spectral_norm_fc


cfg_cifar = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M",],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}

inp_size_cifar = {
    "VGG11": [32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2, 1],
    "VGG13": [32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2, 1],
    "VGG16": [32, 32, 16, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 1],
    "VGG19": [32, 32, 16, 16, 16, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1],
}

cfg_mnist = {
    "VGG11": [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M",],
    "VGG19": [64, 64, 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}

inp_size_mnist = {
    "VGG11": [28, 28, 14, 14, 14, 7, 7, 7, 3, 3, 3, 1],
    "VGG13": [28, 28, 28, 28, 14, 14, 14, 7, 7, 7, 3, 3, 3, 1],
    "VGG16": [28, 28, 28, 28, 14, 14, 14, 14, 7, 7, 7, 7, 3, 3, 3, 3, 1],
    "VGG19": [28, 28, 28, 28, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 1],
}


class VGG(nn.Module):
    def __init__(
        self,
        vgg_name,
        num_classes=10,
        temp=1.0,
        spectral_normalization=True,
        mod=True,
        coeff=3,
        n_power_iterations=1,
        mnist=False,
    ):
        """
        If the "mod" parameter is set to True, the architecture uses 2 modifications:
        1. LeakyReLU instead of normal ReLU
        2. Average Pooling on the residual connections.
        """
        super(VGG, self).__init__()
        self.temp = temp
        self.mod = mod

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            # NOTE: Google uses the spectral_norm_fc in all cases
            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                shapes = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(conv, coeff, shapes, n_power_iterations)

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        self.mnist = mnist
        if mnist:
            self.inp_sizes = inp_size_mnist[vgg_name]
            self.features = self._make_layers(cfg_mnist[vgg_name])
        else:
            self.inp_sizes = inp_size_cifar[vgg_name]
            self.features = self._make_layers(cfg_cifar[vgg_name])

        self.classifier = nn.Linear(512, num_classes)
        self.feature = None

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        self.feature = out.clone().detach()
        out = self.classifier(out) / self.temp
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1 if self.mnist else 3
        for i, x in enumerate(cfg):
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    self.wrapped_conv(self.inp_sizes[i], in_channels, x, kernel_size=3, stride=1),
                    nn.BatchNorm2d(x),
                    nn.LeakyReLU(inplace=True) if self.mod else nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = VGG("VGG11", spectral_normalization=spectral_normalization, mod=mod, temp=temp, mnist=mnist, **kwargs)
    return model


def vgg13(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = VGG("VGG13", spectral_normalization=spectral_normalization, mod=mod, temp=temp, mnist=mnist, **kwargs)
    return model


def vgg16(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = VGG("VGG16", spectral_normalization=spectral_normalization, mod=mod, temp=temp, mnist=mnist, **kwargs)
    return model


def vgg19(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = VGG("VGG19", spectral_normalization=spectral_normalization, mod=mod, temp=temp, mnist=mnist, **kwargs)
    return model
