from typing import List

from torch import nn

from bengali.models.densenet_model import DenseNetBase, DenseNetHead
from bengali.models.resnet_model import ResNetBase, ResNetHead
from bengali.models.vgg_model import VGGBase, VGGHead


def build_model(base: str, n_classes: List[int], **kwargs) -> nn.Module:
    return Model(base=base, n_classes=n_classes, **kwargs)


class Model(nn.Module):
    def __init__(self, *, base: str, n_classes: List[int], **base_kwargs,):
        super().__init__()

        if base.startswith('resne'):

            self.base = ResNetBase(base, **base_kwargs)
            self.in_features = self.base.out_features
            self.grapheme_head = ResNetHead(
                in_features=self.in_features,
                n_classes=n_classes[0],
            )
            self.vowel_head = ResNetHead(
                in_features=self.in_features,
                n_classes=n_classes[1],
            )
            self.consonant_head = ResNetHead(
                in_features=self.in_features,
                n_classes=n_classes[2],
            )
        elif base.startswith('vgg'):
            self.base = VGGBase(base, **base_kwargs)
            self.grapheme_head = VGGHead(
                n_classes=n_classes[0],
            )
            self.vowel_head = VGGHead(
                n_classes=n_classes[1],
            )
            self.consonant_head = VGGHead(
                n_classes=n_classes[2],
            )
        else:
            self.base = DenseNetBase(base, **base_kwargs)
            self.in_features = self.base.out_features
            self.grapheme_head = DenseNetHead(
                in_features=self.in_features,
                n_classes=n_classes[0],
            )
            self.vowel_head = DenseNetHead(
                in_features=self.in_features,
                n_classes=n_classes[1],
            )
            self.consonant_head = DenseNetHead(
                in_features=self.in_features,
                n_classes=n_classes[2],
            )

    def forward(self, x):
        x = self.base(x)
        grapheme = self.grapheme_head(x)
        vowel = self.vowel_head(x)
        consonant = self.consonant_head(x)

        return grapheme, vowel, consonant
