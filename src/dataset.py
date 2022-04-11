import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ToTensor, Normalize, RandomResizedCrop, CenterCrop

from src.transforms import *
from src.autoaugment import AutoAugmentation


def get_train_val_loader(args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    train_transform = Compose([
        RandomResizedCrop(416),
        RandomHorizontalFlip(p=args.fliplr),
        RandomRotation(degrees=args.rot_degree),
        AutoAugmentation(opt=args.autoaugment),
        ToTensor(),
        Normalize(mean, std),
        #GridMask(),
        RandomNoise(p=args.noise)
        
    ])

    val_transform = Compose([
        CenterCrop(416),
        ToTensor(),
        Normalize(mean, std),
    ])

    data_dir = './datasets'
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def get_test_loader(args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_transform = Compose([
        CenterCrop(416),
        ToTensor(),
        Normalize(mean, std),
    ])

    data_dir = './datasets'
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=4)

    return test_loader
    

