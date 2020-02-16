import torch
from torch import nn
from torchvision import models


class ResNetHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.apply_fc_out(x)
        return x

    def apply_fc_out(self, x):
        return self.fc1(x)


class ResNetBase(nn.Module):
    def __init__(self, name: str, pretrained: bool, frozen_start: bool = False):
        super().__init__()

        self.base = getattr(models, name)(pretrained=pretrained)

        self.frozen_start = frozen_start

        if name == 'resnet34' or name == 'resnet18':
            self.out_features = 512
        else:
            self.out_features = 2048

        self.frozen = []
        if self.frozen_start:
            self.frozen = [self.base.layer1, self.base.conv1, self.base.bn1]
            for m in self.frozen:
                self._freeze(m)

    def forward(self, x):
        base = self.base
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)

        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)

        return x

    def train(self, mode=True):
        super().train(mode=mode)
        for m in self.frozen:
            self._bn_to_eval(m)

    def _freeze(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def _bn_to_eval(self, module):
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
