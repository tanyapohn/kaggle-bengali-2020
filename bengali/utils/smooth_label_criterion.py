import torch
from torch import nn
from torch.nn import functional as F


class SmoothLabelCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    2. Calculate loss
    """

    def __init__(self, label_smoothing=0.0):
        super(SmoothLabelCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1)

        # When label smoothing is turned on, KL-divergence is minimized
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            self.criterion = nn.NLLLoss()
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        # Map the output to (0, 1)
        scores = self.LogSoftmax(dec_outs)
        # n_class
        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth)
        return loss


def onehot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)


def cross_entropy_loss(input, target, reduction):
    logp = F.log_softmax(input, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')


def label_smoothing_criterion(epsilon=0.1, reduction='mean'):
    def _label_smoothing_criterion(preds, targets):
        n_classes = preds.size(1)
        device = preds.device

        onehot = onehot_encoding(targets, n_classes).float().to(device)
        targets = (
                onehot * (1 - epsilon) + torch.ones_like(onehot).to(device)
                * epsilon / n_classes
        )
        loss = cross_entropy_loss(preds, targets, reduction)
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(
                '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

    return _label_smoothing_criterion
