from torch import nn

from bengali.grapheme.densenet_model import DenseNetBase, DenseNetHead
from bengali.grapheme.resnet_model import ResNetBase, ResNetHead
from bengali.grapheme.vgg_model import VGGBase, VGGHead


def build_model(base: str, n_classes: int, **kwargs) -> nn.Module:
    return Model(base=base, n_classes=n_classes, **kwargs)


class Model(nn.Module):
    def __init__(self, *, base: str, n_classes: int, **base_kwargs,):
        super().__init__()

        if base.startswith('resnet'):

            self.base = ResNetBase(base, **base_kwargs)
            self.in_features = self.base.out_features
            self.head = ResNetHead(
                in_features=self.in_features,
                n_classes=n_classes,
            )
        elif base.startswith('vgg'):
            self.base = VGGBase(base, **base_kwargs)
            self.head = VGGHead(
                n_classes=n_classes,
            )
        else:
            self.base = DenseNetBase(base, **base_kwargs)
            self.in_features = self.base.out_features
            self.head = DenseNetHead(
                in_features=self.in_features,
                n_classes=n_classes,
            )

    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x
