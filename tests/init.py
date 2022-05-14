"""

"""
import os
from pprint import pformat

import numpy as np
import logging
import json
import sys
from dotenv import load_dotenv, find_dotenv

# Torch
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch import nn


from data import get_loaders

from utils.namers import classifier_params_string, classifier_log_namer


from hahtorch import ImplicitNormalizationConv


def init_logger(cfg, model_name):

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(classifier_log_namer(model_name, cfg)),
            logging.StreamHandler()
            ])

    return logger


def init_tensorboard(cfg, model_name):
    writer = SummaryWriter(cfg.directory + f"tensorboards/{cfg.tensorboard.location}/{cfg.dataset.name}/" +
                           classifier_params_string(model_name, cfg))
    return writer


def init_dataset(cfg):
    train_loader, test_loader = get_loaders(cfg)
    x_min = 0.0
    x_max = 1.0
    data_params = {"x_min": x_min, "x_max": x_max}
    return train_loader, test_loader, data_params



def init_optimizer_scheduler(cfg, model, batches_per_epoch, printer=print, verbose=True):

    if cfg.nn.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.nn.lr,
            momentum=cfg.nn.momentum,
            weight_decay=cfg.nn.weight_decay,
            )
    elif cfg.nn.optimizer == "rms":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=cfg.nn.lr,
            weight_decay=cfg.nn.weight_decay,
            momentum=cfg.nn.momentum,
            )

    elif cfg.nn.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.nn.lr, weight_decay=cfg.nn.weight_decay
            )
    else:
        raise NotImplementedError

    if cfg.nn.scheduler == "cyc":
        lr_steps = cfg.train.epochs * batches_per_epoch
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=cfg.nn.lr_min,
            max_lr=cfg.nn.lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
            )
    elif cfg.nn.scheduler == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(cfg.train.epochs*3/5), int(cfg.train.epochs*4/5)], gamma=0.1
            )

    elif cfg.nn.scheduler == "mult":

        def lr_fun(epoch):
            if epoch % 3 == 0:
                return 0.962
            else:
                return 1.0

        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fun)
    else:
        scheduler = None

    if verbose == True:
        printer(optimizer)
        printer(scheduler)

    return optimizer, scheduler
