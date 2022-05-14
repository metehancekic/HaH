import torch
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
from os import path


def tiny_imagenet(args):

    data_dir = args.data_dir
    train_dir = path.join(data_dir, "original_dataset",
                          "tiny-imagenet-200", "train")
    test_dir = path.join(data_dir, "original_dataset",
                         "tiny-imagenet-200", "val")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]
        )

    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2
        )

    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
        )

    return train_loader, test_loader


def imagenette(args):

    data_dir = args.data_dir
    train_dir = path.join(data_dir, "original_dataset",
                          "imagenette2-160", "train")
    test_dir = path.join(data_dir, "original_dataset",
                         "imagenette2-160", "val")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop((160), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]
        )

    transform_test = transforms.Compose(
        [transforms.CenterCrop(160), transforms.ToTensor()]
        )

    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2
        )

    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
        )

    return train_loader, test_loader


def cifar10(cfg):

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]
        )

    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(
        root=cfg.dataset.directory,
        train=True,
        download=True,
        transform=transform_train,
        )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=2
        )

    testset = datasets.CIFAR10(
        root=cfg.dataset.directory,
        train=False,
        download=True,
        transform=transform_test,
        )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=2
        )

    return train_loader, test_loader


def imagenet(args):

    data_dir = args.data_dir
    train_dir = path.join(data_dir, "original_dataset", "imagenet", "train")
    test_dir = path.join(data_dir, "original_dataset", "imagenet", "val")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ]
        )

    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        )

    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        )

    return train_loader, test_loader


def mnist(cfg):

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if cfg.dataset.name == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                cfg.dataset_dir,
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=cfg.train.batch_size,
            shuffle=True,
            **kwargs
            )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                cfg.dataset_dir,
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=cfg.test.batch_size,
            shuffle=True,
            **kwargs
            )

    elif cfg.dataset.name == "fashion":
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                cfg.dataset_dir,
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=cfg.train.batch_size,
            shuffle=True,
            **kwargs
            )

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                cfg.dataset_dir,
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(), ]),
                ),
            batch_size=cfg.test.batch_size,
            shuffle=True,
            **kwargs
            )

    return train_loader, test_loader


def get_loaders(cfg):

    if cfg.dataset.name == "CIFAR10":
        train_loader, test_loader = cifar10(cfg)
    elif cfg.dataset.name == "Tiny-ImageNet":
        train_loader, test_loader = tiny_imagenet(cfg)
    elif cfg.dataset.name == "Imagenette":
        train_loader, test_loader = imagenette(cfg)
    elif cfg.dataset.name == "Imagenet":
        train_loader, test_loader = imagenet(cfg)
    elif cfg.dataset.name == "mnist":
        train_loader, test_loader = mnist(cfg)
    else:
        raise NotImplementedError

    return train_loader, test_loader
