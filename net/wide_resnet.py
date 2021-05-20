# Obtained from: https://github.com/meliketoy/wide-resnet.pytorch
# Adapted to match:
# https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from net.spectral_normalization.spectral_norm_conv_inplace import spectral_norm_conv
from net.spectral_normalization.spectral_norm_fc import spectral_norm_fc


class WideBasic(nn.Module):
    def __init__(
        self, wrapped_conv, input_size, in_c, out_c, stride, dropout_rate, mod=True, batchnorm_momentum=0.01,
    ):
        super().__init__()

        self.mod = mod
        self.bn1 = nn.BatchNorm2d(in_c, momentum=batchnorm_momentum)
        self.conv1 = wrapped_conv(input_size, in_c, out_c, 3, stride)

        self.bn2 = nn.BatchNorm2d(out_c, momentum=batchnorm_momentum)
        self.conv2 = wrapped_conv(math.ceil(input_size / stride), out_c, out_c, 3, 1)
        self.activation = F.leaky_relu if self.mod else F.relu

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        if stride != 1 or in_c != out_c:
            if mod:

                def shortcut(x):
                    x = F.avg_pool2d(x, stride, stride)
                    pad = torch.zeros(x.shape[0], out_c - in_c, x.shape[2], x.shape[3], device=x.device,)
                    x = torch.cat((x, pad), dim=1)
                    return x

                self.shortcut = shortcut
            else:
                # Just use a strided conv
                self.shortcut = wrapped_conv(input_size, in_c, out_c, 1, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.activation(self.bn1(x))
        out = self.conv1(out)
        out = self.activation(self.bn2(out))

        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = self.conv2(out)
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(
        self,
        spectral_normalization=True,
        mod=True,
        depth=28,
        widen_factor=10,
        num_classes=None,
        dropout_rate=0.3,
        coeff=3,
        n_power_iterations=1,
        batchnorm_momentum=0.01,
        temp=1.0,
        **kwargs
    ):
        """
        If the "mod" parameter is set to True, the architecture uses 2 modifications:
        1. LeakyReLU instead of normal ReLU
        2. Average Pooling on the residual connections.
        """
        super().__init__()

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"

        self.dropout_rate = dropout_rate
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

        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]
        strides = [1, 1, 2, 2]
        input_sizes = 32 // np.cumprod(strides)

        self.conv1 = wrapped_conv(input_sizes[0], 3, nStages[0], 3, strides[0])
        self.layer1 = self._wide_layer(nStages[0:2], n, strides[1], input_sizes[0])
        self.layer2 = self._wide_layer(nStages[1:3], n, strides[2], input_sizes[1])
        self.layer3 = self._wide_layer(nStages[2:4], n, strides[3], input_sizes[2])

        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=batchnorm_momentum)
        self.activation = F.leaky_relu if self.mod else F.relu

        self.num_classes = num_classes
        if num_classes is not None:
            self.linear = nn.Linear(nStages[3], num_classes)

        nonlinearity = "leaky_relu" if self.mod else "relu"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Sergey implementation has no mode/nonlinearity
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L17
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            elif isinstance(m, nn.Linear):
                # Sergey implementation has no mode/nonlinearity
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L21
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
                nn.init.constant_(m.bias, 0)
        self.feature = None
        self.temp = temp

    def _wide_layer(self, channels, num_blocks, stride, input_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        in_c, out_c = channels

        for stride in strides:
            layers.append(WideBasic(self.wrapped_conv, input_size, in_c, out_c, stride, self.dropout_rate, self.mod,))
            in_c = out_c
            input_size = math.ceil(input_size / stride)

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.activation(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.flatten(1)
        self.feature = out.clone().detach()

        if self.num_classes is not None:
            out = self.linear(out) / self.temp
        return out


def wrn(temp=1.0, spectral_normalization=True, mod=True, **kwargs):
    model = WideResNet(spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)
    return model
