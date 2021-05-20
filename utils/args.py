"""
Contains common args used in different scripts.
"""

import argparse


def training_args():

    default_dataset = "cifar10"
    dataset_root = "./"
    ood_dataset = "svhn"
    train_batch_size = 128
    test_batch_size = 128

    learning_rate = 0.1
    momentum = 0.9
    optimiser = "sgd"
    loss = "cross_entropy"
    weight_decay = 5e-4
    log_interval = 50
    save_interval = 25
    save_loc = "./"
    saved_model_name = "resnet50_350.model"
    epoch = 350
    first_milestone = 150  # Milestone for change in lr
    second_milestone = 250  # Milestone for change in lr

    model = "resnet50"
    sn_coeff = 3.0

    parser = argparse.ArgumentParser(
        description="Args for training parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument(
        "--dataset", type=str, default=default_dataset, dest="dataset", help="dataset to train on",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=dataset_root,
        dest="dataset_root",
        help="path of a dataset (useful for dirty mnist)",
    )
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)

    parser.add_argument(
        "-b", type=int, default=train_batch_size, dest="train_batch_size", help="Batch size",
    )
    parser.add_argument(
        "-tb", type=int, default=test_batch_size, dest="test_batch_size", help="Test Batch size",
    )

    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Use GPU")
    parser.add_argument("--model", type=str, default=model, dest="model", help="Model to train")

    parser.add_argument(
        "-sn", action="store_true", dest="sn", help="whether to use spectral normalisation during training",
    )
    parser.set_defaults(sn=False)
    parser.add_argument(
        "--coeff", type=float, default=sn_coeff, dest="coeff", help="Coeff parameter for spectral normalisation",
    )
    parser.add_argument(
        "-mod", action="store_true", dest="mod", help="whether to use architectural modifications during training",
    )
    parser.set_defaults(mod=False)

    parser.add_argument("-e", type=int, default=epoch, dest="epoch", help="Number of training epochs")
    parser.add_argument(
        "--lr", type=float, default=learning_rate, dest="learning_rate", help="Learning rate",
    )
    parser.add_argument("--mom", type=float, default=momentum, dest="momentum", help="Momentum")
    parser.add_argument(
        "--nesterov", action="store_true", dest="nesterov", help="Whether to use nesterov momentum in SGD",
    )
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        "--decay", type=float, default=weight_decay, dest="weight_decay", help="Weight Decay",
    )
    parser.add_argument(
        "--opt", type=str, default=optimiser, dest="optimiser", help="Choice of optimisation algorithm",
    )

    parser.add_argument(
        "--loss", type=str, default=loss, dest="loss_function", help="Loss function to be used for training",
    )
    parser.add_argument(
        "--loss-mean",
        action="store_true",
        dest="loss_mean",
        help="whether to take mean of loss instead of sum to train",
    )
    parser.set_defaults(loss_mean=False)

    parser.add_argument(
        "--log-interval", type=int, default=log_interval, dest="log_interval", help="Log Interval on Terminal",
    )
    parser.add_argument(
        "--save-interval", type=int, default=save_interval, dest="save_interval", help="Save Interval on Terminal",
    )
    parser.add_argument(
        "--saved_model_name",
        type=str,
        default=saved_model_name,
        dest="saved_model_name",
        help="file name of the pre-trained model",
    )
    parser.add_argument(
        "--save-path", type=str, default=save_loc, dest="save_loc", help="Path to export the model",
    )

    parser.add_argument(
        "--first-milestone",
        type=int,
        default=first_milestone,
        dest="first_milestone",
        help="First milestone to change lr",
    )
    parser.add_argument(
        "--second-milestone",
        type=int,
        default=second_milestone,
        dest="second_milestone",
        help="Second milestone to change lr",
    )

    return parser


def eval_args():
    default_dataset = "cifar10"
    ood_dataset = "svhn"
    batch_size = 128
    load_loc = "../Models/Normal/"
    model = "resnet50"
    sn_coeff = 3.0
    runs = 5
    ensemble = 5
    model_type = "softmax"

    parser = argparse.ArgumentParser(
        description="Training for calibration.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument(
        "--dataset", type=str, default=default_dataset, dest="dataset", help="dataset to train on",
    )
    parser.add_argument(
        "--ood_dataset",
        type=str,
        default=ood_dataset,
        dest="ood_dataset",
        help="OOD dataset for given training dataset",
    )
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=False)

    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-b", type=int, default=batch_size, dest="batch_size", help="Batch size")
    parser.add_argument(
        "--load-path", type=str, default=load_loc, dest="load_loc", help="Path to load the model from",
    )
    parser.add_argument("--model", type=str, default=model, dest="model", help="Model to train")
    parser.add_argument(
        "--runs", type=int, default=runs, dest="runs", help="Number of models to aggregate over",
    )

    parser.add_argument(
        "-sn", action="store_true", dest="sn", help="whether to use spectral normalisation during training",
    )
    parser.set_defaults(sn=False)
    parser.add_argument(
        "--coeff", type=float, default=sn_coeff, dest="coeff", help="Coeff parameter for spectral normalisation",
    )
    parser.add_argument(
        "-mod", action="store_true", dest="mod", help="whether to use architectural modifications during training",
    )
    parser.set_defaults(mod=False)
    parser.add_argument("--ensemble", type=int, default=ensemble, dest="ensemble", help="Number of models in ensemble")
    parser.add_argument(
        "--model-type",
        type=str,
        default=model_type,
        choices=["softmax", "ensemble", "gmm"],
        dest="model_type",
        help="Type of model to load for evaluation.",
    )

    return parser


def al_args():
    model_name = "resnet18"
    trained_model_name = "resnet18_sn"
    saved_model_path = "./"
    saved_model_name = "resnet18_sn_3.0_50.model"
    dataset_root = "./"
    threshold = 1.0
    subsample = 1000
    al_acquisition = "softmax"

    sn_coeff = 3.0
    num_ensemble = 5

    num_initial_samples = 20
    max_training_samples = 300
    acquisition_batch_size = 5
    epochs = 20

    train_batch_size = 64
    test_batch_size = 512
    scoring_batch_size = 128

    parser = argparse.ArgumentParser(description="Active Learning Experiments")
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument(
        "--model", type=str, default=model_name, dest="model_name", help="Model to train",
    )
    parser.add_argument(
        "-ambiguous", action="store_true", dest="ambiguous", help="Use Ambiguous MNIST in training",
    )
    parser.set_defaults(ambiguous=False)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=dataset_root,
        dest="dataset_root",
        help="path of a dataset (useful for ambiguous mnist)",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        default=trained_model_name,
        dest="trained_model_name",
        help="Trained model to check entropy of acquired samples",
    )

    parser.add_argument(
        "-tsn", action="store_true", dest="tsn", help="whether to use spectral normalisation",
    )
    parser.set_defaults(tsn=False)
    parser.add_argument(
        "--tcoeff", type=float, default=sn_coeff, dest="tcoeff", help="Coeff parameter for spectral normalisation",
    )
    parser.add_argument(
        "-tmod", action="store_true", dest="tmod", help="whether to use architectural modifications during training",
    )
    parser.set_defaults(tmod=False)

    parser.add_argument(
        "--saved-model-path",
        type=str,
        default=saved_model_path,
        dest="saved_model_path",
        help="Path of pretrained model",
    )
    parser.add_argument(
        "--saved-model-name",
        type=str,
        default=saved_model_name,
        dest="saved_model_name",
        help="File name of pretrained model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=threshold,
        dest="threshold",
        help="Entropy threshold to decide if a sample is ambiguous or not",
    )
    parser.add_argument(
        "--subsample", type=int, default=subsample, dest="subsample", help="Subsample for MNIST",
    )

    parser.add_argument(
        "--num-ensemble", type=int, default=num_ensemble, dest="num_ensemble", help="Number of models in ensemble",
    )

    parser.add_argument(
        "-sn", action="store_true", dest="sn", help="whether to use spectral normalisation during training",
    )
    parser.set_defaults(sn=False)
    parser.add_argument(
        "--coeff", type=float, default=sn_coeff, dest="coeff", help="Coeff parameter for spectral normalisation",
    )
    parser.add_argument(
        "-mod", action="store_true", dest="mod", help="whether to use architectural modifications during training",
    )
    parser.set_defaults(mod=False)

    parser.add_argument(
        "--al-type",
        type=str,
        default=al_acquisition,
        choices=["softmax", "ensemble", "gmm"],
        dest="al_type",
        help="Type of model to use for AL.",
    )

    parser.add_argument("-mi", action="store_true", dest="mi", help="Use MI as acquisition function")
    parser.set_defaults(mi=False)

    parser.add_argument(
        "--num-initial-samples",
        type=int,
        default=num_initial_samples,
        dest="num_initial_samples",
        help="Initial number of samples in the training set",
    )
    parser.add_argument(
        "--max-training-samples",
        type=int,
        default=max_training_samples,
        dest="max_training_samples",
        help="Maximum training set size",
    )
    parser.add_argument(
        "--acquisition-batch-size",
        type=int,
        default=acquisition_batch_size,
        dest="acquisition_batch_size",
        help="Number of samples to acquire in each acquisition step",
    )

    parser.add_argument(
        "-e", type=int, default=epochs, dest="epochs", help="Number of epochs to train after each acquisition",
    )
    parser.add_argument(
        "-b", type=int, default=train_batch_size, dest="train_batch_size", help="Training batch size",
    )
    parser.add_argument(
        "-tb", type=int, default=test_batch_size, dest="test_batch_size", help="Test batch size",
    )
    parser.add_argument(
        "-sb",
        type=int,
        default=scoring_batch_size,
        dest="scoring_batch_size",
        help="Batch size for scoring the pool dataset",
    )

    return parser
