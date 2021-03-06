from torch import nn

from bengali.models.densenet_model import DenseNetBase, DenseNetHead
from bengali.models.resnet_model import ResNetBase, ResNetHead
from bengali.models.vgg_model import VGGBase, VGGHead
from bengali.models.senet import SENetBase, SENetHead


def build_model(base: str, n_classes: int, **kwargs) -> nn.Module:
    return Model(base=base, n_classes=n_classes, **kwargs)


class Model(nn.Module):
    def __init__(self, *, base: str, n_classes: int, **base_kwargs,):
        super().__init__()

        # TODO: Put spatial transform here

        if base.startswith('resne'):

            self.base = ResNetBase(base, **base_kwargs)
            self.in_features = self.base.out_features
            self.head = ResNetHead(
                in_features=self.in_features,
                n_classes=n_classes,
            )
        elif base.startswith('se_'):

            self.base = SENetBase(base, **base_kwargs)
            self.in_features = self.base.out_features
            self.head = SENetHead(
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
