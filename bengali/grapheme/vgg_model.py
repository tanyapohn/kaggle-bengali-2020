import torch
from torch import nn
from torchvision import models


class VGGHead(nn.Module):

    def __init__(
            self, n_classes: int, in_features: int = 25088,
            dropout: float = 0.5, hidden_dim: int = 4096,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(hidden_dim, n_classes, bias=True),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class VGGBase(nn.Module):

    def __init__(self, name: str, pretrained: bool = True):

        super().__init__()
        self.base = getattr(models, name)(pretrained=pretrained)

    def forward(self, x):
        base = self.base
        x = base.features(x)
        x = base.avgpool(x)
        x = torch.flatten(x, 1)
        return x
