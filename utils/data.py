# data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import medmnist as med
from medmnist import INFO
import torchvision.transforms as transforms
import numpy as np

class BinaryLabelDataset(Dataset):
    def __init__(self, dataset, target_label, info):
        self.dataset = dataset
        self.target_label = dict(zip(info["label"].values(), info["label"].keys()))[target_label]
        self.target_label = int(self.target_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, labels = self.dataset[idx]
        binary_label = 1 if self.target_label in labels else 0
        binary_label = np.array([binary_label])
        return sample, torch.tensor(binary_label)


def get_data_loader(args):
    batch_size = args.batch_size if hasattr(args,"batch_size") else 64
    data_flag = args.data_flag if hasattr(args, "data_flag") else 'pathmnist'
    info = INFO[data_flag]

    # assumes all datasets have same number of channels
    n_channels = info['n_channels']
    n_classes = 2 if args.label_to_binary else None
    label_dict = {}
    download = True

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Get dataset splits
    DataClass = getattr(med, info['python_class'])
    train_dataset = DataClass(split="train", transform=data_transform, download=True)
    val_dataset = DataClass(split="val", transform=data_transform, download=True)
    test_dataset = DataClass(split="test", transform=data_transform, download=True)

    # if label given, set label to 1 where present, 0 otherwise
    if args.label_to_binary is not None:
        label_dict = {
            "0": args.label_to_binary,
            "1": f"not_{args.label_to_binary}"     
        }
        train_dataset = BinaryLabelDataset(train_dataset, args.label_to_binary, info)
        val_dataset = BinaryLabelDataset(val_dataset, args.label_to_binary, info)
        test_dataset = BinaryLabelDataset(test_dataset, args.label_to_binary, info)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # hardcoded for now

    params = {'n_channels': n_channels, 'n_classes': n_classes, "label_dict": label_dict}

    return train_loader, val_loader, test_loader, params


def get_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
