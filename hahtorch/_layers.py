from typing import Union, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from math import floor


class ImplicitNormalizationConv(nn.Conv2d):
    r"""
    torch.Conv2d with implicit weight normalization (L2 normalization)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_norms = (self.weight**2).sum(dim=(1, 2, 3),
                                            keepdim=True).transpose(0, 1).sqrt()

        conv = super().forward(x)
        return conv/(weight_norms+1e-6)


class Normalize(nn.Module):
    r"""Data normalizing class as torch.nn.Module

    Args:
        mean (float): Mean value of the training dataset.
        std (float): Standard deviation value of the training dataset.

    """

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        """
        Args:
            x (tensor batch): Input tensor.
        Returns:
            Normalized data.
        """
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class DivisiveNormalization2d(nn.Module):
    r"""Applies a 2D divisive normalization over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`
    and output :math:`(N, C, H, W)`.

    Args:
        b_type: Type of suppressin field, must be one of (`linf`, `l1`, `l2`).
        b_size: The size of the suppression field, must be > 0.
        sigma: Constant added to suppression field, must be > 0.
        alpha: Global suppression rate, 0 means no suppression, 1 means complete suppression.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`

    Examples::

        >>> # suppression of size=3, sigma=1
        >>> d = DivisiveNormalization2d(b_size=3, sigma=1)
        >>> input = torch.randn(20, 16, 50, 50)
        >>> output = d(input)
    """

    def __init__(
            self,
            b_type: str = "linf",
            b_size: Union[int, Tuple[int, int]] = (3, 3),
            sigma: float = 1.0,
            alpha: float = 0.0,
            ) -> None:
        super(DivisiveNormalization2d, self).__init__()

        self.sigma = sigma
        self.alpha = alpha

        if isinstance(b_size, int):
            self.b_size = (b_size, b_size)
        else:
            self.b_size = b_size
        self.padding = (self.b_size[0]//2, self.b_size[1]//2)
        self.b_type = b_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.b_type == "linf":
            suppression_field = F.max_pool2d(
                torch.abs(input), self.b_size, 1, self.padding, 1)
        elif self.b_type == "l1":
            weight = torch.ones(
                (input.shape[1], 1, self.b_size[0], self.b_size[1])).to(input.device)
            suppression_field = F.conv2d(
                torch.abs(input), weight=weight, padding=self.padding, groups=input.shape[1])
        elif self.b_type == "l2":
            weight = torch.ones(
                (input.shape[1], 1, self.b_size[0], self.b_size[1])).to(input.device)
            suppression_field = torch.sqrt(F.conv2d(
                input**2, weight=weight, padding=self.padding, groups=input.shape[1]))
        else:
            raise NotImplementedError

        suppression_field = torch.mean(suppression_field, dim=1, keepdim=True)
        divisor = (self.sigma*torch.amax(suppression_field, dim=(2, 3),
                                         keepdim=True) + (1-self.sigma) * suppression_field + 0.0000001)
        return input / divisor

    def __repr__(self) -> str:
        s = "DivisiveNormalization2d("
        s += f'b_type={self.b_type}, b_size={self.b_size}, sigma={self.sigma}'
        if self.alpha > 0:
            s += f', alpha={self.alpha}'
        s += ")"
        return s


class AdaptiveThreshold(nn.Module):
    r"""
    Thresholds values x[x>threshold]
    Args:
        mean_scalar: threshold/means
    """

    def __init__(self, mean_scalar: float = 1.0) -> None:
        super(AdaptiveThreshold, self).__init__()

        self.mean_scalar = mean_scalar

    def _thresholding(self, x, threshold):
        return x*(x > threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        means = input.mean(dim=(2, 3), keepdim=True)
        return self._thresholding(input, means*self.mean_scalar)

    def __repr__(self) -> str:
        s = f"AdaptiveThreshold(mean_scalar={self.mean_scalar})"
        return s


class FractionalThreshold(nn.Module):
    r"""
    FractionalThreshold 
    Args:
        remaining_ratio: Remaining activation ratio after Threshold.
    """

    def __init__(self, remaining_ratio: float = 0.1) -> None:
        super(FractionalThreshold, self).__init__()

        self.remaining_ratio = remaining_ratio

    def _thresholding(self, x: torch.Tensor, threshold: Union[torch.Tensor, float]) -> torch.Tensor:
        return x*(x > threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        k = floor((1-self.remaining_ratio)*input.shape[2]*input.shape[3])
        thresholds = torch.kthvalue(input.view(
            *input.shape[:2], -1), k, dim=-1, keepdim=True)[0].unsqueeze(-1)
        return self._thresholding(input, thresholds)

    def __repr__(self) -> str:
        s = f"FractionalThreshold(remaining_ratio={self.remaining_ratio})"
        return s


class HaHBlock(nn.Module):
    r"""
    FractionalThreshold 
    Args:
        in_channels: Input channels for convolutional layer
        out_channels: Output channels for convolutional layer
        kernel_size: Kernel Size for convolutional layer
        padding: Padding for convolutional layer
        divisive_sigma: Hyperparameter for Divisive normalization
        ratio: Thresholding ratio (remaining_ratio)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, divisive_sigma=0.1, ratio=0.2) -> None:
        super(HaHBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.divisive_sigma = divisive_sigma
        self.ratio = ratio

        self.conv_layer = ImplicitNormalizationConv(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False)
        self.relu = torch.nn.ReLU(inplace=False)
        self.normalization_layer = DivisiveNormalization2d(sigma=self.divisive_sigma)
        self.threshold_layer = FractionalThreshold(remaining_ratio=self.ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv_layer(x)
        o = self.relu(o)
        o = self.normalization_layer(o)
        o = self.threshold_layer(o)
        return o

    def __repr__(self) -> str:
        s = f"HaHBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding}, divisive_sigma={self.divisive_sigma}, ratio={self.ratio})"
        return s
