"""
Script for training deep ensemble models.
"""

import torch
import argparse
from torch import optim
import torch.backends.cudnn as cudnn

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn

# Import network models
from net.resnet import resnet50
from net.wide_resnet import wrn
from net.vgg import vgg16

# Import train and validation utilities
from utils.args import training_args
from utils.train_utils import train_single_epoch, test_single_epoch
from utils.train_utils import model_save_name
from utils.eval_utils import get_eval_stats_ensemble

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter


dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10}

dataset_loader = {"cifar10": cifar10, "cifar100": cifar100, "svhn": svhn}

models = {"resnet50": resnet50, "wide_resnet": wrn, "vgg16": vgg16}


def parseArgs():
    ensemble = 5
    parser = training_args()
    parser.add_argument(
        "--ensemble", type=int, default=ensemble, dest="ensemble", help="Number of ensembles to train",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parseArgs()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net_ensemble = [
        models[args.model](spectral_normalization=args.sn, mod=args.mod, coeff=args.coeff, num_classes=num_classes,).to(
            device
        )
        for _ in range(args.ensemble)
    ]

    optimizers = []
    schedulers = []
    train_loaders = []

    for i, model in enumerate(net_ensemble):
        opt_params = model.parameters()
        if args.optimiser == "sgd":
            optimizer = optim.SGD(
                opt_params,
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov,
            )
        elif args.optimiser == "adam":
            optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1,
        )
        train_loader, _ = data_set_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size, augment=args.data_aug, val_size=0.1, val_seed=args.seed+i, pin_memory=args.gpu,
        )
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        train_loaders.append(train_loader)

    # Creating summary writer in tensorboard
    writer = SummaryWriter(args.save_loc + "stats_logging/")

    for epoch in range(0, args.epoch):
        for i, model in enumerate(net_ensemble):
            print("Ensemble Model: " + str(i))
            train_loss = train_single_epoch(
                epoch, model, train_loaders[i], optimizers[i], device, loss_mean=args.loss_mean,
            )
            schedulers[i].step()

        writer.add_scalar(args.model + "_ensemble_" + "train_loss", train_loss, (epoch + 1))
        save_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
        if (epoch + 1) % args.save_interval == 0:
            for i, model in enumerate(net_ensemble):
                save_name = args.save_loc + save_name + str(args.seed+i) + "_" + str(epoch + 1) + ".model"
                torch.save(model.state_dict(), save_name)
    writer.close()
