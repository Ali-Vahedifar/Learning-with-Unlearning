from typing import List, Optional, Type

import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 100,
        input_channels: int = 3,
        small_input: bool = True,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        if small_input:
            self.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: Type[BasicBlock], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.get_features(x))


def resnet18(num_classes: int = 100, small_input: bool = True, **kwargs) -> ResNet:
    return ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes, small_input=small_input, **kwargs
    )


class FPECNN(nn.Module):
    def __init__(self, num_classes: int = 10, flatten_size: int = 256, hidden_size: int = 64):
        super().__init__()
        self.flatten_size = flatten_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        def block(in_channels, out_channels):
            layers = []
            for index in range(3):
                layers += [
                    nn.Conv2d(
                        in_channels if index == 0 else out_channels,
                        out_channels,
                        3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            return layers

        self.conv_layers = nn.Sequential(
            *block(3, 64),
            nn.MaxPool2d(2),
            *block(64, 128),
            nn.MaxPool2d(2),
            *block(128, 256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, flatten_size, 1, bias=False),
            nn.BatchNorm2d(flatten_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc1 = nn.Linear(flatten_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

    def forward(self, inputs):
        return self.forward_from_features(self.get_features(inputs))

    def forward_from_features(self, features):
        return self.fc2(self.relu(self.fc1(features)))

    def get_features(self, inputs):
        return self.conv_layers(inputs).flatten(1)


BACKBONES = {'resnet18': resnet18, 'fpecnn': FPECNN}


def get_backbone(name: str, num_classes: int, **kwargs) -> nn.Module:
    if name not in BACKBONES:
        raise ValueError(f'unknown backbone {name!r}; available: {sorted(BACKBONES)}')
    return BACKBONES[name](num_classes=num_classes, **kwargs)
