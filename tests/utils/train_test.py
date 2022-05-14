"""
Description: Training and testing functions for neural models

functions:
    train: Performs a single training epoch (if attack_args is present adversarial training)
    test: Evaluates model by computing accuracy (if attack_args is present adversarial testing)
"""

from tqdm import tqdm
import numpy as np
from functools import partial
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F


from .augmentation import get_noisy_images

from hahtorch import HaHCost


def single_epoch(cfg,
                 model,
                 train_loader,
                 optimizer,
                 scheduler=None,
                 compute_activation_disparity=True,
                 logging_func=print,
                 verbose: bool = True,
                 epoch: int = 0):
    r"""
    Single epoch
    """
    start_time = time()
    model.train()
    device = model.parameters().__next__().device

    cross_ent = nn.CrossEntropyLoss()
    train_loss = 0
    train_correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        if "hah" in cfg.train.regularizer.active:
            hah_cost = HaHCost(ratio=cfg.train.regularizer.hah.ratio,
                               demote=cfg.train.regularizer.hah.lamda,
                               dim=cfg.train.regularizer.hah.dim)
            _ = model(data)
            hah_loss_per_layer: list[torch.Tensor] = []
            for idx, (layer, layer_input, layer_output) in enumerate(zip(model.layers_of_interest.values(), model.layer_inputs.values(), model.layer_outputs.values())):
                loss_args = dict(
                    ratio=cfg.train.regularizer.hah.ratio,
                    saliency_lambda=cfg.train.regularizer.hah.lamda,
                    dim=cfg.train.regularizer.hah.dim)

                hah_loss_ = hah_cost(layer_output)

                hah_loss_per_layer.append(hah_loss_)

            scalar_vector = torch.Tensor(cfg.train.regularizer.hah.alpha).to(device)
            for idx, (layer, layer_input, layer_output) in list(enumerate(zip(model.layers_of_interest.values(), model.layer_inputs.values(), model.layer_outputs.values())))[::-1]:
                if scalar_vector[idx] == 0.0:
                    continue
                if layer.weight.grad is not None:
                    layer.weight.grad.zero_()

                layer_hah_loss = -scalar_vector[idx] * \
                    hah_loss_per_layer[idx]
                layer_hah_loss.backward(retain_graph=True)

        # Noisy Training
        if cfg.train.type == "noisy":
            data = get_noisy_images(data, cfg.train.noise.std)

        # Adversarial Training
        if cfg.train.type == "adversarial":
            from deepillusion.torchattacks import FGSM, RFGSM, PGD
            perturbs = locals()[cfg.train.adversarial.attack](net=model, x=data, y_true=target, data_params={
                "x_min": cfg.dataset.min, "x_max": cfg.dataset.max}, attack_params=cfg.train.adversarial, verbose=False)
            data += perturbs

        loss = 0.0
        output = model(data)
        xent_loss = cross_ent(output, target)

        l1_weight_loss = 0
        if "l1_weight" in cfg.train.regularizer.active:
            for _, layer in model.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    l1_weight_loss += l1_loss(
                        features={"conv": layer.weight}, dim=(1, 2, 3))

        loss += xent_loss + cfg.train.regularizer.l1_weight.scale * l1_weight_loss

        loss.backward()
        optimizer.step()
        if scheduler and cfg.nn.scheduler == "cyc":
            scheduler.step()

        train_loss += xent_loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=False)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    if scheduler and not cfg.nn.scheduler == "cyc":
        scheduler.step()

    train_size = len(train_loader.dataset)
    train_loss = train_loss/train_size
    train_acc = train_correct/train_size

    if verbose:
        logging_func(
            f"Epoch: \t {epoch} \t Time (s): {(time()-start_time):.0f}")
        logging_func(
            f"Train Xent loss: \t {train_loss:.2f} \t Train acc: {100*train_acc:.2f} %")
        logging_func(f"L1 Weight Loss: \t {l1_weight_loss:.4f}")
        logging_func("-"*100)


def standard_test(model, test_loader, verbose=True, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)
    if verbose:
        print(
            f"Test loss: {test_loss/test_size:.4f}, Test acc: {100*test_correct/test_size:.2f}")

    return test_loss/test_size, test_correct/test_size
