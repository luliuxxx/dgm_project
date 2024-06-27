# data
import torch
import torch.nn as nn
import medmnist as med
from medmnist import INFO
import torchvision.transforms as transforms
import numpy as np


def get_data_loader(args = None):
    batch_size = args.batch_size if args else 32
    data_flag = args.data_flag if args else 'pathmnist'

    info = INFO[data_flag]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    download = False

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    DataClass = getattr(med, info['python_class'])

    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

    params = {'n_channels': n_channels, 'n_classes': n_classes}

    return train_loader, val_loader, test_loader, params


def get_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params