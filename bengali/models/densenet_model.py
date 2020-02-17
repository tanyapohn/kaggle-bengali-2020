import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class DenseNetHead(nn.Module):

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.classifier(x)


class DenseNetBase(nn.Module):

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        self.base = getattr(models, name)(pretrained=pretrained)
        if name.endswith('121'):
            self.out_features = 1024
        elif name.endswith('161'):
            self.out_features = 2208
        elif name.endswith('169'):
            self.out_features = 1664
        else:
            self.out_features = 1920

    def forward(self, x):
        base = self.base
        features = base.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out
