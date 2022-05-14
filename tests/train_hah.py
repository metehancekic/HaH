"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import os

import torch
from torch.nn import Conv2d

# PYTORCH UTILS
from utils.namers import classifier_ckpt_namer
from utils.train_test import single_epoch, standard_test
from utils.analysis import count_parameter

from hahtorch import ActivationExtractor
from hahtorch import ImplicitNormalizationConv
from hahtorch import HaHVGG16, VGG16

# Initializers
from init import *


@hydra.main(config_path="/home/metehan/HaH/tests/configs", config_name="cifar")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, _ = init_dataset(cfg)
    model = globals()[cfg.nn.classifier]().to(device)

    model = ActivationExtractor(model, layer_type=globals()[cfg.train.regularizer.hah.layer])
    logger = init_logger(cfg, model.name)

    if cfg.verbose:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(model)
        logger.info(f"Model will be saved to {classifier_ckpt_namer(model_name=model.name, cfg=cfg)}")

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.nn.lr, weight_decay=cfg.nn.weight_decay
        )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(cfg.train.epochs*3/5), int(cfg.train.epochs*4/5)], gamma=0.1
        )

    _ = count_parameter(model=model, logger=logger.info, verbose=cfg.verbose)

    for epoch in range(1, cfg.train.epochs+1):
        single_epoch(cfg=cfg, model=model, train_loader=train_loader,
                     optimizer=optimizer, scheduler=scheduler, verbose=True, epoch=epoch)
        if epoch % cfg.log_interval == 0 or epoch == cfg.train.epochs:
            _, __ = standard_test(model=model, test_loader=test_loader,
                                  verbose=True, progress_bar=False)

    if cfg.save_model:
        os.makedirs(cfg.directory + "checkpoints/classifiers/", exist_ok=True)
        classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
        torch.save(model.state_dict(), classifier_filepath)
        logger.info(f"Saved to {classifier_filepath}")


if __name__ == "__main__":
    main()
