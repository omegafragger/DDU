import torch
from torch import nn
from tqdm import tqdm

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    check_once = True
    with torch.no_grad():
        start = 0
        # for data, label in tqdm(loader):
        print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
        index = 0
        for data, label in loader:
            index += 1
            print("----------Inside---------")
            image_data = [datum for datum in data]
            image_data = torch.stack(image_data, dim=0)
            image_data = image_data.to(device)
            # data = data.to(device)
            # label = label.to(device)
            labels_data = [label_val for label_val in label]
            labels_data = torch.tensor(labels_data)
            labels_data = labels_data.to(device)

            # if isinstance(net, nn.DataParallel):
            #     out = net.module(data)
            #     out = net.module.feature
            # else:
            #     out = net(data)
            #     if check_once:
            #         print("Forward Pass Output:", out)
            #     out = net.feature
            #     if check_once:
            #         check_once = False
            print("CHECK ONCE:", check_once)
            if isinstance(net, nn.DataParallel):
                if check_once:
                    print("Net Feature Before:", net.module.feature)
                out = net.module(image_data)
                if check_once:
                    print("Output:", out.shape)
                    print("Output:", out)
                    print("Net Feature After:", net.module.feature)
                    print("Net Shape:", net.module.feature.shape)
                    check_once = False
                out = net.module.feature
            else:
                if check_once:
                    print("Net Feature:", net.feature)
                out = net(image_data)
                if check_once:
                    print(out.shape)
                out = net.feature
                if check_once:
                    print("Net Feature:", net.feature)
                    print("Net Shape:", net.feature.shape)
                check_once = False
            # end = start + len(data)
            end = start + len(image_data)
            embeddings[start:end].copy_(out, non_blocking=True)
            # labels[start:end].copy_(label, non_blocking=True)
            labels[start:end].copy_(labels_data, non_blocking=True)
            print("Index:", index)
            print(embeddings.shape)
            print(labels.shape)
            start = end
        print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print("Embeddings Shape:", embeddings.shape)
    print("Labels Shape:", labels.shape)
    return embeddings, labels


def gmm_forward(net, gaussians_model, data_B_X):

    if isinstance(net, nn.DataParallel):
        features_B_Z = net.module(data_B_X)
        features_B_Z = net.module.feature
    else:
        features_B_Z = net(data_B_X)
        features_B_Z = net.feature

    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            logit_B_C = gmm_forward(net, gaussians_model, data)
            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_compute_logits(net, gaussians_model, loader, device, num_classes, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for image_data, labels_data, image_names in tqdm(loader):
            data = [image for image in image_data]
            data = torch.stack(data, dim=0)
            labels = [label for label in labels_data]
            labels = torch.tensor(labels)
            data = data.to(device)
            labels = labels.to(device)

            logit_B_C = gmm_forward(net, gaussians_model, data)
            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            start = end
    return logits_N_C


def gmm_get_logits(gmm, embeddings):

    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
    return log_probs_B_Y


def gmm_fit(embeddings, labels, num_classes):
    with torch.no_grad():
        print("Embeddings Shape:", embeddings.shape, "Labels Shape:", labels.shape, "Num Classes:", num_classes)
        classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
        # Dim - (4 * 1024)
        print("Classwise Mean Features Shape:", classwise_mean_features.shape)
        classwise_cov_features = torch.stack(
            [centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)]
        )
        # Dim - (4 * 1024 * 1024)
        print("Classwise Covariance Matrix Shape:", classwise_cov_features.shape)
    # gmm = None
    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1], device=classwise_cov_features.device,
                ).unsqueeze(0)
                # print("Jitter Shape:", jitter.shape)
                # print("#"*50, "YES", "#"*50)
                gmm = torch.distributions.MultivariateNormal(
                      loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
                )
                print("Jitter Succeeded:", jitter_eps)
                print("#"*50, "GMM:", gmm, "#"*50)
            except RuntimeError as e:
                print("Jitter Failed:", jitter_eps)
                # print("Exception 1:", e)
                # if "cholesky" in str(e):
                #     continue
                continue
            except ValueError as e:
                print("Jitter Failed:", jitter_eps)
                # print("Exception 2:", e)
                # if "The parameter covariance_matrix has invalid values" in str(e):
                    # print(e)
                    # continue
                continue
            except Exception as e:
                print("Jitter Failed:", jitter_eps)
                # print(e)
                continue
            # break
            break
    return gmm, jitter_eps
