import json
import torch
import numpy as np
import torch.backends.cudnn as cudnn

# Import data utilities
import torch.utils.data as data
import data.active_learning.active_learning as active_learning
from data.ambiguous_mnist.ambiguous_mnist_dataset import AmbiguousMNIST
from data.fast_mnist import create_MNIST_dataset

# Import network architectures
from net.resnet import resnet18

# Import train and test utils
from utils.train_utils import train_single_epoch, model_save_name

# Importing uncertainty metrics
from metrics.uncertainty_confidence import entropy, logsumexp, confidence
from metrics.classification_metrics import test_classification_net
from metrics.classification_metrics import test_classification_net_ensemble

# Importing args
from utils.args import al_args

# Importing GMM utilities
from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit
from utils.ensemble_utils import ensemble_forward_pass


# Mapping model name to model function
models = {"resnet18": resnet18}


def class_probs(data_loader):
    num_classes = 10
    class_n = len(data_loader.dataset)
    class_count = torch.zeros(num_classes)
    for data, label in data_loader:
        class_count += torch.Tensor([torch.sum(label == c) for c in range(num_classes)])

    class_prob = class_count / class_n
    return class_prob


def compute_density(logits, class_probs):
    return torch.sum((torch.exp(logits) * class_probs), dim=1)


def ambiguous_acquired(data_loader, threshold, model):
    """
    This method is required to identify the ambiguous samples which are acquired.
    """
    model.eval()
    logits = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            op = model(data)
            logits.append(op)

        logits = torch.cat(logits, dim=0)
    entropies = entropy(logits)

    return entropies.cpu().numpy().tolist(), (torch.sum(entropies > threshold).item() / len(data_loader.dataset))


if __name__ == "__main__":

    args = al_args().parse_args()
    print(args)

    # Checking if GPU is available
    cuda = torch.cuda.is_available()

    # Setting additional parameters
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if cuda else "cpu")

    model_fn = models[args.model_name]

    # Load pretrained network for checking ambiguous samples
    if args.ambiguous:
        pretrained_net = models[args.trained_model_name](spectral_normalization=args.tsn, mod=args.tmod, mnist=True).to(device)
        pretrained_net = torch.nn.DataParallel(pretrained_net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        pretrained_net.load_state_dict(torch.load(args.saved_model_path + args.saved_model_name))

    # Creating the datasets
    num_classes = 10
    train_dataset, test_dataset = create_MNIST_dataset()
    if args.ambiguous:
        indices = np.random.choice(len(train_dataset), args.subsample)
        mnist_train_dataset = data.Subset(train_dataset, indices)
        train_dataset = data.ConcatDataset(
            [mnist_train_dataset, AmbiguousMNIST(root=args.dataset_root, train=True, device=device),]
        )

    # Creating a validation split
    idxs = list(range(len(train_dataset)))
    split = int(np.floor(0.1 * len(train_dataset)))
    np.random.seed(args.seed)
    np.random.shuffle(idxs)

    train_idx, val_idx = idxs[split:], idxs[:split]
    val_dataset = data.Subset(train_dataset, val_idx)
    train_dataset = data.Subset(train_dataset, train_idx)

    initial_sample_indices = active_learning.get_balanced_sample_indices(
        train_dataset, num_classes=num_classes, n_per_digit=args.num_initial_samples / num_classes,
    )

    kwargs = {"num_workers": 0, "pin_memory": False} if cuda else {}
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Run experiment
    num_runs = 5
    test_accs = {}
    ambiguous_dict = {}
    ambiguous_entropies_dict = {}

    for i in range(num_runs):
        test_accs[i] = []
        ambiguous_dict[i] = []
        ambiguous_entropies_dict[i] = {}

    for run in range(num_runs):
        print("Experiment run: " + str(run) + " =====================================================================>")

        torch.manual_seed(args.seed + run)

        # Setup data for the experiment
        # Split off the initial samples first
        active_learning_data = active_learning.ActiveLearningData(train_dataset)

        # Acquiring the first training dataset from the total pool. This is random acquisition
        active_learning_data.acquire(initial_sample_indices)

        # Train loader for the current acquired training set
        sampler = active_learning.RandomFixedLengthSampler(
            dataset=active_learning_data.training_dataset, target_length=5056
        )
        train_loader = torch.utils.data.DataLoader(
            active_learning_data.training_dataset, sampler=sampler, batch_size=args.train_batch_size, **kwargs,
        )

        small_train_loader = torch.utils.data.DataLoader(
            active_learning_data.training_dataset, shuffle=True, batch_size=args.train_batch_size, **kwargs,
        )

        # Pool loader for the current acquired training set
        pool_loader = torch.utils.data.DataLoader(
            active_learning_data.pool_dataset, batch_size=args.scoring_batch_size, shuffle=False, **kwargs,
        )

        # Run active learning iterations
        active_learning_iteration = 0
        while True:
            print("Active Learning Iteration: " + str(active_learning_iteration) + " ================================>")

            lr = 0.1
            weight_decay = 5e-4
            if args.al_type == "ensemble":
                model_ensemble = [
                    model_fn(spectral_normalization=args.sn, mod=args.mod, mnist=True).to(device=device)
                    for _ in range(args.num_ensemble)
                ]
                optimizers = []
                for model in model_ensemble:
                    optimizers.append(torch.optim.Adam(model.parameters(), weight_decay=weight_decay))
                    model.train()
            else:
                model = model_fn(spectral_normalization=args.sn, mod=args.mod, mnist=True).to(device=device)
                optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
                model.train()

            # Train
            print("Length of train dataset: " + str(len(train_loader.dataset)))
            best_model = None
            best_val_accuracy = 0
            for epoch in range(args.epochs):
                if args.al_type == "ensemble":
                    for (model, optimizer) in zip(model_ensemble, optimizers):
                        train_single_epoch(epoch, model, train_loader, optimizer, device)
                else:
                    train_single_epoch(epoch, model, train_loader, optimizer, device)

                _, val_accuracy, _, _, _ = (
                    test_classification_net_ensemble(model_ensemble, val_loader, device=device)
                    if args.al_type == "ensemble"
                    else test_classification_net(model, val_loader, device=device)
                )
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model = model_ensemble if args.al_type == "ensemble" else model

            if args.al_type == "ensemble":
                model_ensemble = best_model
            else:
                model = best_model

            if args.al_type == "gmm":
                # Fit the GMM on the trained model
                model.eval()
                embeddings, labels = get_embeddings(
                    model, small_train_loader, num_dim=512, dtype=torch.double, device="cuda", storage_device="cuda",
                )
                gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)
            print("Training ended")

            # Testing the models
            if args.al_type == "ensemble":
                print("Testing the model: Ensemble======================================>")
                for model in model_ensemble:
                    model.eval()
                (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net_ensemble(
                    model_ensemble, test_loader, device=device
                )

            else:
                print("Testing the model: Softmax/GMM======================================>")
                (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net(
                    model, test_loader, device=device
                )
            percentage_correct = 100.0 * accuracy
            test_accs[run].append(percentage_correct)

            print("Test set: Accuracy: ({:.2f}%)".format(percentage_correct))

            # Breaking clause
            if len(active_learning_data.training_dataset) >= args.max_training_samples:
                break

            # Acquisition phase
            N = len(active_learning_data.pool_dataset)

            print("Performing acquisition ========================================")
            if args.al_type == "ensemble":
                for model in model_ensemble:
                    model.eval()
                ensemble_uncs = []
                with torch.no_grad():
                    for data, _ in pool_loader:
                        data = data.to(device)
                        mean_output, predictive_entropy, mi = ensemble_forward_pass(model_ensemble, data)

                        ensemble_uncs.append(mi if args.mi else predictive_entropy)
                    ensemble_uncs = torch.cat(ensemble_uncs, dim=0)

                    (candidate_scores, candidate_indices,) = active_learning.get_top_k_scorers(
                        ensemble_uncs, args.acquisition_batch_size
                    )
            else:
                model.eval()
                if args.al_type == "gmm":
                    class_prob = class_probs(train_loader)
                    logits, labels = gmm_evaluate(
                        model,
                        gaussians_model,
                        pool_loader,
                        device=device,
                        num_classes=num_classes,
                        storage_device="cpu",
                    )
                    (candidate_scores, candidate_indices,) = active_learning.get_top_k_scorers(
                        compute_density(logits, class_prob), args.acquisition_batch_size, uncertainty=False,
                    )
                else:
                    logits = []
                    with torch.no_grad():
                        for data, _ in pool_loader:
                            data = data.to(device)
                            logits.append(model(data))
                        logits = torch.cat(logits, dim=0)
                    (candidate_scores, candidate_indices,) = active_learning.find_acquisition_batch(
                        logits, args.acquisition_batch_size, entropy
                    )

            # Performing acquisition
            active_learning_data.acquire(candidate_indices)
            if args.ambiguous:
                entropies, amb_percent = ambiguous_acquired(small_train_loader, args.threshold, pretrained_net)
                ambiguous_dict[run].append(amb_percent)
                ambiguous_entropies_dict[run][active_learning_iteration] = entropies
            active_learning_iteration += 1

    # Save the dictionaries
    save_name = model_save_name(args.model_name, args.sn, args.mod, args.coeff, args.seed)
    save_ensemble_mi = "_mi" if (args.al_type == "ensemble" and args.mi) else ""
    if args.ambiguous:
        accuracy_file_name = (
            "test_accs_" + save_name + '_' + args.al_type + save_ensemble_mi + "_dirty_mnist_" + str(args.subsample) + ".json"
        )
        ambiguous_file_name = (
            "ambiguous_" + save_name + '_' + args.al_type + save_ensemble_mi + "_dirty_mnist_" + str(args.subsample) + ".json"
        )
        ambiguous_entropies_file_name = (
            "ambiguous_entropies_" + save_name + '_' + args.al_type + save_ensemble_mi + "_dirty_mnist_" + str(args.subsample) + ".json"
        )
    else:
        accuracy_file_name = "test_accs_" + save_name + '_' + args.al_type + save_ensemble_mi + "_mnist.json"

    with open(accuracy_file_name, "w") as acc_file:
        json.dump(test_accs, acc_file)

    if args.ambiguous:
        with open(ambiguous_file_name, "w") as ambiguous_file:
            json.dump(ambiguous_dict, ambiguous_file)
        with open(ambiguous_entropies_file_name, "w") as ambiguous_entropies_file:
            json.dump(ambiguous_entropies_dict, ambiguous_entropies_file)
