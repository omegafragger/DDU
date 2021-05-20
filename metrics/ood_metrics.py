# Utility functions to get OOD detection ROC curves and AUROC scores
# Ideally should be agnostic of model architectures

import torch
import torch.nn.functional as F
from sklearn import metrics

from utils.ensemble_utils import ensemble_forward_pass
from metrics.classification_metrics import get_logits_labels
from metrics.uncertainty_confidence import entropy, logsumexp, confidence


def get_roc_auc(net, test_loader, ood_test_loader, uncertainty, device, confidence=False):
    logits, _ = get_logits_labels(net, test_loader, device)
    ood_logits, _ = get_logits_labels(net, ood_test_loader, device)

    return get_roc_auc_logits(logits, ood_logits, uncertainty, device, confidence=confidence)


def get_roc_auc_logits(logits, ood_logits, uncertainty, device, confidence=False):
    uncertainties = uncertainty(logits)
    ood_uncertainties = uncertainty(ood_logits)

    # In-distribution
    bin_labels = torch.zeros(uncertainties.shape[0]).to(device)
    in_scores = uncertainties

    # OOD
    bin_labels = torch.cat((bin_labels, torch.ones(ood_uncertainties.shape[0]).to(device)))

    if confidence:
        bin_labels = 1 - bin_labels
    ood_scores = ood_uncertainties  # entropy(ood_logits)
    scores = torch.cat((in_scores, ood_scores))

    fpr, tpr, thresholds = metrics.roc_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
    precision, recall, prc_thresholds = metrics.precision_recall_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
    auroc = metrics.roc_auc_score(bin_labels.cpu().numpy(), scores.cpu().numpy())
    auprc = metrics.average_precision_score(bin_labels.cpu().numpy(), scores.cpu().numpy())

    return (fpr, tpr, thresholds), (precision, recall, prc_thresholds), auroc, auprc


def get_roc_auc_ensemble(model_ensemble, test_loader, ood_test_loader, uncertainty, device):
    bin_labels_uncertainties = None
    uncertainties = None

    for model in model_ensemble:
        model.eval()

    bin_labels_uncertainties = []
    uncertainties = []
    with torch.no_grad():
        # Getting uncertainties for in-distribution data
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            bin_label_uncertainty = torch.zeros(label.shape).to(device)
            if uncertainty == "mutual_information":
                net_output, _, unc = ensemble_forward_pass(model_ensemble, data)
            else:
                net_output, unc, _ = ensemble_forward_pass(model_ensemble, data)

            bin_labels_uncertainties.append(bin_label_uncertainty)
            uncertainties.append(unc)

        # Getting entropies for OOD data
        for data, label in ood_test_loader:
            data = data.to(device)
            label = label.to(device)

            bin_label_uncertainty = torch.ones(label.shape).to(device)
            if uncertainty == "mutual_information":
                net_output, _, unc = ensemble_forward_pass(model_ensemble, data)
            else:
                net_output, unc, _ = ensemble_forward_pass(model_ensemble, data)

            bin_labels_uncertainties.append(bin_label_uncertainty)
            uncertainties.append(unc)

        bin_labels_uncertainties = torch.cat(bin_labels_uncertainties)
        uncertainties = torch.cat(uncertainties)

    fpr, tpr, roc_thresholds = metrics.roc_curve(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())
    precision, recall, prc_thresholds = metrics.precision_recall_curve(
        bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy()
    )
    auroc = metrics.roc_auc_score(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())
    auprc = metrics.average_precision_score(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())

    return (fpr, tpr, roc_thresholds), (precision, recall, prc_thresholds), auroc, auprc
