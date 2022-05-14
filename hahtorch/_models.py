from typing import Union, Tuple, Iterable

import torch
import torch.nn.functional as F
import torch.nn as nn

from ._layers import Normalize, HaHBlock

from math import floor


vgg_depth_dict = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGBlock(nn.Module):
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

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1) -> None:
        super(VGGBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv_layer = torch.nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False)
        self.relu = torch.nn.ReLU(inplace=False)
        self.normalization_layer = nn.BatchNorm2d(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv_layer(x)
        o = self.relu(o)
        o = self.normalization_layer(o)
        return o

    def __repr__(self) -> str:
        s = f"VGGBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding})"
        return s


class HaHVGG16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normalize = Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2471, 0.2435, 0.2616))
        self.vgg_depth = "VGG16"
        self.hah_layer_num = 7

        self.features = self.make_layers(vgg_depth_dict[self.vgg_depth])
        self.classifier = nn.Linear(512, 10)

        # self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, layer_list: Iterable[Union[str, int]]) -> nn.Sequential:
        blocks: list[nn.Module] = []
        in_channels = 3
        for layer_i, layer_info in enumerate(layer_list):
            if isinstance(layer_info, str):
                blocks += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layer_width = layer_info

                if layer_i <= self.hah_layer_num:
                    blocks += [HaHBlock(in_channels=in_channels, out_channels=layer_width,
                                      kernel_size=3, padding=1, divisive_sigma=0.1, ratio=0.2)]
                else:
                    blocks += [VGGBlock(in_channels=in_channels, out_channels=layer_width,
                                      kernel_size=3, padding=1)]

                in_channels = layer_width

        return nn.Sequential(*blocks)

    @property
    def name(self) -> str:
        s = "HaHVGG16"
        return s


class VGG16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normalize = Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2471, 0.2435, 0.2616))
        self.vgg_depth = "VGG16"
        self.hah_layer_num = 7

        self.features = self.make_layers(vgg_depth_dict[self.vgg_depth])
        self.classifier = nn.Linear(512, 10)

        # self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, layer_list: Iterable[Union[str, int]]) -> nn.Sequential:
        blocks: list[nn.Module] = []
        in_channels = 3
        for layer_i, layer_info in enumerate(layer_list):
            if isinstance(layer_info, str):
                blocks += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layer_width = layer_info

                blocks = +[VGGBlock(in_channels=in_channels, out_channels=layer_width,
                                  kernel_size=3, padding=1)]

                in_channels = layer_width

        return nn.Sequential(*blocks)

    @property
    def name(self) -> str:
        s = "VGG16"
        return s
