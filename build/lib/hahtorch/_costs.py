from typing import Dict, Union, Tuple, Optional
import numpy as np
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = ["HaHCost", "L1Cost"]


class _Cost(nn.Module):
    """
    Base cost class
    Args:
        dim: dimension where reduction is applied (Generally channel dimension)
        reduction: Reduction type
    """

    def __init__(self, dim: int = 1, reduction: str = "mean") -> None:
        super(_Cost, self).__init__()
        self.dim = dim
        self.reduction = reduction


class HaHCost(_Cost):
    """
    Hebbian/Anti-Hebbian Cost class
    Args:
        ratio: Top ratio to be promoted
        demote: Hyperparameter determining how much to emphasize the anti-Hebbian component
        dim: dimension where reduction is applied (Generally channel dimension)
        reduction: Reduction type
    """

    def __init__(self, ratio: float = 0.1, demote: float = 1.0, dim: int = 1, reduction: str = "mean") -> None:
        super(HaHCost, self).__init__(dim, reduction)
        self.ratio = ratio
        self.demote = demote

    def forward(self, input: Tensor) -> Tensor:

        N = input.shape[self.dim]
        K = ceil(self.ratio * N)

        o = F.relu(input)
        o = torch.sort(o, dim=self.dim, descending=True)[0]

        top_K_avg = o[:, :K].mean(dim=self.dim)
        bottom_avg = o[:, K:].mean(dim=self.dim)

        if self.reduction == "mean":
            return torch.mean(top_K_avg-self.demote*bottom_avg)
        elif self.reduction == "sum":
            return torch.sum(top_K_avg-self.demote*bottom_avg)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        s = f"HaHCost(ratio={self.ratio}, demote={self.demote}, dim={self.dim}, reduction={self.reduction})"
        return s


class L1Cost(_Cost):
    """
    L1 Cost class
    Args:
        dim: dimension where reduction is applied (Generally channel dimension)
        reduction: Reduction type
    """

    def __init__(self, dim: int = 1, reduction: str = "mean") -> None:
        super(HaHCost, self).__init__(dim, reduction)

    def forward(self, input: Tensor, ratio: float) -> Tensor:
        return torch.mean(torch.sum(torch.abs(input), dim=self.dim))

    def __repr__(self) -> str:
        s = f"L1Cost(dim={self.dim}, reduction={self.reduction})"
        return s
