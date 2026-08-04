"""
Backbone Neural Network Architectures

Implements ResNet-18 and other architectures for use with the LwU framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type, Union


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion: int = 1
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet architecture for image classification."""
    
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 100,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Type[nn.Module]] = None,
        input_channels: int = 3,
        small_input: bool = True
    ) -> None:
        """
        Initialize ResNet.
        
        Args:
            block: Block type (BasicBlock for ResNet-18)
            layers: Number of blocks in each layer
            num_classes: Number of output classes
            zero_init_residual: Zero-initialize the last BN in each residual branch
            groups: Number of groups for grouped convolution
            width_per_group: Width of each group
            norm_layer: Normalization layer to use
            input_channels: Number of input channels
            small_input: Use smaller input configuration (for CIFAR/TinyImageNet)
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        # Modified first layers for smaller inputs (CIFAR-100, TinyImageNet)
        if small_input:
            self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialization (He initialization)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        """Create a layer with multiple blocks."""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer
        ))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer
            ))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classification layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def expand_output_layer(self, new_classes: int):
        """
        Expand the output layer to accommodate new classes.
        
        Args:
            new_classes: Number of new classes to add
        """
        old_fc = self.fc
        old_classes = old_fc.out_features
        new_total = old_classes + new_classes
        
        # Create new FC layer
        new_fc = nn.Linear(old_fc.in_features, new_total)
        
        # Copy old weights
        new_fc.weight.data[:old_classes] = old_fc.weight.data
        new_fc.bias.data[:old_classes] = old_fc.bias.data
        
        # Initialize new weights with He initialization
        nn.init.kaiming_normal_(new_fc.weight.data[old_classes:])
        nn.init.zeros_(new_fc.bias.data[old_classes:])
        
        self.fc = new_fc


def resnet18(num_classes: int = 100, small_input: bool = True, **kwargs) -> ResNet:
    """Construct a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, 
                  small_input=small_input, **kwargs)


def resnet34(num_classes: int = 100, small_input: bool = True, **kwargs) -> ResNet:
    """Construct a ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                  small_input=small_input, **kwargs)


class SimpleMLP(nn.Module):
    """Simple MLP for tabular/1D data (e.g., RTI dataset)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        num_classes: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_dim, num_classes)
        
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
    
    def expand_output_layer(self, new_classes: int):
        """Expand output layer for new classes."""
        old_fc = self.fc
        old_classes = old_fc.out_features
        new_total = old_classes + new_classes
        
        new_fc = nn.Linear(old_fc.in_features, new_total)
        new_fc.weight.data[:old_classes] = old_fc.weight.data
        new_fc.bias.data[:old_classes] = old_fc.bias.data
        
        nn.init.kaiming_normal_(new_fc.weight.data[old_classes:])
        nn.init.zeros_(new_fc.bias.data[old_classes:])
        
        self.fc = new_fc


class SimpleCNN(nn.Module):
    """Simple CNN for smaller datasets or faster experimentation."""
    
    def __init__(
        self,
        num_classes: int = 100,
        input_channels: int = 3
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(256, num_classes)
        
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
    
    def expand_output_layer(self, new_classes: int):
        """Expand output layer for new classes."""
        old_fc = self.fc
        old_classes = old_fc.out_features
        new_total = old_classes + new_classes
        
        new_fc = nn.Linear(old_fc.in_features, new_total)
        new_fc.weight.data[:old_classes] = old_fc.weight.data
        new_fc.bias.data[:old_classes] = old_fc.bias.data
        
        nn.init.kaiming_normal_(new_fc.weight.data[old_classes:])
        nn.init.zeros_(new_fc.bias.data[old_classes:])
        
        self.fc = new_fc


def get_backbone(
    name: str,
    num_classes: int,
    input_dim: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create backbone networks.
    
    Args:
        name: Name of the backbone ('resnet18', 'resnet34', 'mlp', 'simplecnn')
        num_classes: Number of output classes
        input_dim: Input dimension (required for MLP)
        **kwargs: Additional arguments for the backbone
        
    Returns:
        Neural network module
    """
    def _vit_b16_lora():
        from lwu.models.vit_lora import vit_b16_lora
        return vit_b16_lora(num_classes=num_classes, **kwargs)

    backbones = {
        'resnet18': lambda: resnet18(num_classes=num_classes, **kwargs),
        'resnet34': lambda: resnet34(num_classes=num_classes, **kwargs),
        'mlp': lambda: SimpleMLP(input_dim=input_dim, num_classes=num_classes, **kwargs),
        'simplecnn': lambda: SimpleCNN(num_classes=num_classes, **kwargs),
        'vit_b16_lora': _vit_b16_lora,
        'vit_b16': _vit_b16_lora
    }
    
    if name not in backbones:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(backbones.keys())}")
        
    return backbones[name]()
