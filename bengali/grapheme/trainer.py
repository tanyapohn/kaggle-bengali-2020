from typing import Callable, Tuple

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from bengali.grapheme.cutmix import cutmix, cutmix_criterion


def train_one_epoch(
        data_train_loader: DataLoader, model: Module,
        mixed_prob: float, fp16: bool, device,
        loss: Callable, optimizer,
) -> Tuple:

    cum_grapheme_loss = 0
    cum_accuracy = 0

    model.train()
    for idx, (input_image, label) in tqdm(
            enumerate(data_train_loader), total=len(data_train_loader),
    ):
        input_image = input_image.to(device, dtype=torch.float)
        grapheme_label = label.to(device, dtype=torch.long)

        r = np.random.rand()
        alpha = np.random.uniform(low=0.8, high=1.)

        # if MIXED:
        if r < mixed_prob:
            input_image, targets = cutmix(
                input_image, grapheme_label, alpha=alpha,
            )
            pred_grapheme = model(input_image)
            grapheme_loss = cutmix_criterion(
                pred_grapheme, targets,
            )
        else:
            pred_grapheme = model(input_image)
            grapheme_loss = loss(pred_grapheme, grapheme_label)

        optimizer.zero_grad()

        if fp16:
            from apex import amp
            with amp.scale_loss(grapheme_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            grapheme_loss.backward()

        optimizer.step()

        cum_grapheme_loss += grapheme_loss.item()
        cum_accuracy += (pred_grapheme.argmax(1) == grapheme_label).float().mean().item()

    n_total = len(data_train_loader)
    epoch_loss = cum_grapheme_loss / n_total
    epoch_acc = cum_accuracy / n_total

    return epoch_loss, epoch_acc
