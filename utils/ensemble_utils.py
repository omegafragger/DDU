"""
Utilities for processing a deep ensemble.
"""
import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from net.vgg import vgg16
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.wide_resnet import wrn

from metrics.uncertainty_confidence import entropy_prob, mutual_information_prob


models = {
    "lenet": lenet,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet": wrn,
    "vgg16": vgg16,
}


def load_ensemble(ensemble_loc, model_name, device, num_classes=10, ensemble_len=5, num_epochs=350, seed=1, **kwargs):
    ensemble = []
    cudnn.benchmark = True
    for i in range(ensemble_len):
        net = models[model_name](num_classes=num_classes, temp=1.0, **kwargs).to(device)
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net.load_state_dict(
            torch.load(ensemble_loc + '/' + model_name + '_' + str(seed+i) + "_" + str(num_epochs) + ".model")
        )
        ensemble.append(net)
    return ensemble


def ensemble_forward_pass(model_ensemble, data):
    """
    Single forward pass in a given ensemble providing softmax distribution,
    predictive entropy and mutual information.
    """
    outputs = []
    for i, model in enumerate(model_ensemble):
        output = F.softmax(model(data), dim=1)
        outputs.append(torch.unsqueeze(output, dim=0))

    outputs = torch.cat(outputs, dim=0)
    mean_output = torch.mean(outputs, dim=0)
    predictive_entropy = entropy_prob(mean_output)
    mut_info = mutual_information_prob(outputs)

    return mean_output, predictive_entropy, mut_info
