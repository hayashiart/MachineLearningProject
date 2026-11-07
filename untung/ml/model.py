"""MobileNetV3-Large backbone + new 38-class classifier head.

Transfer learning per the project plan: load ImageNet-pretrained weights and
replace only the final linear layer.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    # classifier = Dropout -> Linear(1280,1280) -> Hardswish -> Linear(1280, C)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad = True


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
