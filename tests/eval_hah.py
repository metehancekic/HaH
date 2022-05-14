"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

from cgi import test
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import os

# ATTACK CODES
from robustbench.data import load_cifar10
from autoattack import AutoAttack

# Initializers
from init import *

from utils import standard_test, test_noisy
from utils.namers import classifier_ckpt_namer


from hahtorch import ActivationExtractor
from hahtorch import ImplicitNormalizationConv
from hahtorch import HaHVGG16, VGG16

@hydra.main(config_path="/home/metehan/HaH/tests/configs", config_name="cifar")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    _, test_loader, __ = init_dataset(cfg)
    model = globals()[cfg.nn.classifier]().to(device)

    logger = init_logger(cfg, model.name)

    classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
    model.load_state_dict(torch.load(classifier_filepath))

    model = ActivationExtractor(model, layer_type=torch.nn.Conv2d)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if cfg.verbose:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(model)
        logger.info(classifier_filepath)

    test_loss, test_acc = standard_test(
        model=model, test_loader=test_loader, verbose=False, progress_bar=False)
    logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    # Noisy Test Set Evaluation
    noisy_acc = [None]*cfg.noisy.num_experiments
    noisy_loss = [None]*cfg.noisy.num_experiments
    for i in range(cfg.noisy.num_experiments):
        noise_std = cfg.noisy.std
        noisy_acc[i], noisy_loss[i], _ = test_noisy(model, test_loader, noise_std)

    logger.info(f'Noise std 0.1: Test  \t loss: {sum(noisy_loss)/cfg.noisy.num_experiments:.4f} \t acc: {sum(noisy_acc)/cfg.noisy.num_experiments:.4f}')

    # Auto Attack
    x_test, y_test = load_cifar10(n_examples=10000, data_dir=cfg.dataset.directory)

    adversary = AutoAttack(model, norm='Linf', eps=2/255, version="custom",
                           attacks_to_run=['apgd-ce', 'apgd-t'])
    adversary.apgd.n_restarts = 1

    print("#" + "-"*100 + "#")
    print("-"*40 + " L_inf " + "-"*40)
    print("#" + "-"*100 + "#")

    x_adv = adversary.run_standard_evaluation(x_test, y_test)


if __name__ == "__main__":
    main()
