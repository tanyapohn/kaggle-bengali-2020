from typing import List

from torch import nn

from bengali.models.densenet_model import DenseNetBase, DenseNetHead
from bengali.models.resnet_model import ResNetBase, ResNetHead
from bengali.models.transformation import TPS_SpatialTransformerNetwork
from bengali.models.vgg_model import VGGBase, VGGHead


def build_model(base: str, n_classes: List[int], **kwargs) -> nn.Module:
    return Model(base=base, n_classes=n_classes, **kwargs)


class Model(nn.Module):
    def __init__(self, *, base: str, n_classes: List[int], **base_kwargs,):
        super().__init__()

        self.transformation = TPS_SpatialTransformerNetwork(
            F=20,
            I_size=(224, 224),
            I_r_size=(224, 224),
            I_channel_num=3,
        )
        if base.startswith('resne'):

            self.base = ResNetBase(base, **base_kwargs)
            self.in_features = self.base.out_features
            self.vowel_head = ResNetHead(
                in_features=self.in_features,
                n_classes=n_classes[0],
            )
            self.consonant_head = ResNetHead(
                in_features=self.in_features,
                n_classes=n_classes[1],
            )
        elif base.startswith('vgg'):
            self.base = VGGBase(base, **base_kwargs)
            self.vowel_head = VGGHead(
                n_classes=n_classes[0],
            )
            self.consonant_head = VGGHead(
                n_classes=n_classes[1],
            )
        else:
            self.base = DenseNetBase(base, **base_kwargs)
            self.in_features = self.base.out_features
            self.vowel_head = DenseNetHead(
                in_features=self.in_features,
                n_classes=n_classes[0],
            )
            self.consonant_head = DenseNetHead(
                in_features=self.in_features,
                n_classes=n_classes[1],
            )

    def forward(self, x):
        x = self.base(x)
        vowel = self.vowel_head(x)
        consonant = self.consonant_head(x)

        return vowel, consonant
