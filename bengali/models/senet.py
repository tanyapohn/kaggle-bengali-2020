from torch import nn
import pretrainedmodels


class SENetHead(nn.Module):

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.last_linear = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


class SENetBase(nn.Module):

    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        self.pretrained = pretrained
        self.base = pretrainedmodels.__dict__[name](
            num_classes=1000, pretrained='imagenet'
        )
        self.out_features = 2048

    def forward(self, x):
        base = self.base
        features = base.features(x)

        return features
