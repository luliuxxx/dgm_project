# data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import medmnist as med
from medmnist import INFO
import torchvision.transforms as transforms
import numpy as np

class CombinedMedMNISTDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for dataset_idx, dataset in enumerate(self.datasets):
            if idx < len(dataset):
                data, _ = dataset[idx]
                return data, dataset_idx
            idx -= len(dataset)
        raise IndexError("Index out of range in CombinedMedMNISTDataset")


def get_data_loader(args = None):
    batch_size = args.batch_size if args else 32
    data_flag = args.data_flag if args else 'pathmnist'

    # ensure data_flag is list
    if not isinstance(data_flag, list):
        data_flag = [data_flag]

    # assumes all datasets have same number of channels
    info = INFO[data_flag[0]]
    n_channels = info['n_channels']
    n_classes = len(data_flag)
    download = True

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Handle single or multiple datasets
    def load_datasets(split):
        datasets = []
        for flag in data_flag:
            info = INFO[flag]
            DataClass = getattr(med, info['python_class'])
            dataset = DataClass(split=split, transform=data_transform, download=True)
            datasets.append(dataset)
        return CombinedMedMNISTDataset(datasets) if len(datasets) > 1 else datasets[0]

    train_dataset = load_datasets('train')
    val_dataset = load_datasets('val')
    test_dataset = load_datasets('test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    params = {'n_channels': n_channels, 'n_classes': n_classes}

    return train_loader, val_loader, test_loader, params


def get_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
