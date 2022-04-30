import torch
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import glob as glob
import cv2
import os

class DDUDataset(Dataset):
	def __init__(self, data_dir_path, train_image_dir, data_label_json, width, height, transforms=None):
		self.train_label_path = os.path.join(data_dir_path, data_label_json)
		self.vehicle_class_dictionary = None
		with open(self.train_label_path, "r") as fp:
			self.vehicle_class_dictionary = json.load(fp)
		self.transforms = transforms
		self.width = width
		self.height = height
		self.train_image_dir_path = os.path.join(data_dir_path, train_image_dir)
		self.image_paths = glob.glob(f"{self.train_image_dir_path}/*.png")
		self.all_images = [image_path.split('/')[-1].split('.')[0] for image_path in self.image_paths]
		self.total_possible_classes = len(set(self.vehicle_class_dictionary.values()))

	def __getitem__(self, idx):
		train_image_name = self.all_images[idx]
		train_image_label = self.vehicle_class_dictionary[train_image_name]
		# print("Train Image Par Dir Path, Name, Label:", self.train_image_dir_path, train_image_name, train_image_label)
		train_image_path = os.path.join(self.train_image_dir_path, train_image_name)
		train_image_path = train_image_path + ".png"
		# print("Train Image Path:", train_image_path)
		train_image = cv2.imread(train_image_path)
		# print("Shape:", train_image.shape)
		train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB).astype(np.float32)
		train_image_resized = cv2.resize(train_image, (self.width, self.height))
		# print("Shape:", train_image_resized.shape)
		train_image_orig_width = train_image.shape[1]
		train_image_orig_height = train_image.shape[0]
		train_image_label_tensor = torch.as_tensor(int(train_image_label), dtype=torch.int64)
		train_image_tensor = torch.tensor(train_image_resized).permute(2,0,1)
		# print(idx, train_image_name, train_image_label_tensor)
		return train_image_tensor, train_image_label_tensor

	def __len__(self):
		return len(self.all_images)


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def get_train_valid_loader(batch_size=None, val_seed=None, val_size=0.1, num_workers=4, pin_memory=False, root=None, **kwargs):
    error_msg = "[!] val_size should be in the range [0, 1]."
    assert (val_size >= 0) and (val_size <= 1), error_msg

    # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)

    # # define transforms
    # valid_transform = transforms.Compose([transforms.ToTensor(), normalize,])

    # if augment:
    #     train_transform = transforms.Compose(
    #         [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]
    #     )
    # else:
    #     train_transform = transforms.Compose([transforms.ToTensor(), normalize,])

    # load the dataset
    # data_dir = "./data"
    # train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform,)

    # valid_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=valid_transform,)

    data_dir_path = "data"
    data_label_json = "vehicle_class_dict.json"
    train_image_dir = "cropped_images"
    torch_dataset = DDUDataset(data_dir_path, train_image_dir, data_label_json, 32, 32)

    # num_train = len(train_dataset)
    num_train = len(torch_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.seed(val_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    # train_subset = Subset(train_dataset, train_idx)
    # valid_subset = Subset(valid_dataset, valid_idx)
    train_subset = Subset(torch_dataset, train_idx)
    valid_subset = Subset(torch_dataset, valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn, shuffle=True 
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_subset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn, shuffle=False
    )
    return (train_loader, valid_loader)


def get_test_loader(batch_size, num_workers=4, pin_memory=False, **kwargs):
    # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)
    # # define transform
    # transform = transforms.Compose([transforms.ToTensor(), normalize,])
    # data_dir = "./data"
    # dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform,)

	data_dir_path = "data"
	data_label_json = "vehicle_class_dict.json"
	train_image_dir = "cropped_images"

	torch_dataset = DDUDataset(data_dir_path, train_image_dir, data_label_json, 32, 32)

	data_loader = torch.utils.data.DataLoader(
	   torch_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
	)

	return data_loader