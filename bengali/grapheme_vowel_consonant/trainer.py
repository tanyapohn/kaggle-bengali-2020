from typing import Callable, Tuple

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from bengali.grapheme_vowel_consonant.cutmix import cutmix, cutmix_criterion


def train_one_epoch(
        data_train_loader: DataLoader, model: Module, mixed_prob: float,
        fp16: bool, device, loss: Callable, optimizer,
) -> Tuple:

    total_loss = 0
    grapheme_accuracy = 0
    vowel_accuracy = 0
    consonant_accuracy = 0

    model.train()
    for idx, (input_image, labels) in tqdm(
            enumerate(data_train_loader), total=len(data_train_loader),
    ):
        input_image = input_image.to(device, dtype=torch.float)
        grapheme_label = labels[:, 0].to(device, dtype=torch.long)
        vowel_label = labels[:, 1].to(device, dtype=torch.long)
        consonant_label = labels[:, 2].to(device, dtype=torch.long)

        r = np.random.rand()
        alpha = np.random.uniform(low=0.8, high=1.)

        # if MIXED:
        if r < mixed_prob:
            input_image, targets = cutmix(
                input_image, grapheme_label, vowel_label,
                consonant_label, alpha=alpha,
            )
            pred_grapheme, pred_vowel, pred_consonant = model(input_image)

            losses = cutmix_criterion(
                pred_grapheme, pred_vowel,
                pred_consonant, targets,
            )
        else:
            pred_grapheme, pred_vowel, pred_consonant = model(input_image)

            loss_grapheme = loss(pred_grapheme, grapheme_label)
            loss_vowel = loss(pred_vowel, vowel_label)
            loss_consonant = loss(pred_consonant, consonant_label)

            losses = 0.5*loss_grapheme + 0.25*loss_vowel + 0.25*loss_consonant

        optimizer.zero_grad()

        if fp16:
            from apex import amp
            with amp.scale_loss(losses, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses.backward()

        optimizer.step()

        total_loss += losses.item()

        grapheme_accuracy += (pred_grapheme.argmax(1) == grapheme_label).float().mean().item()
        vowel_accuracy += (pred_vowel.argmax(1) == vowel_label).float().mean().item()
        consonant_accuracy += (pred_consonant.argmax(1) == consonant_label).float().mean().item()

    total_acc = grapheme_accuracy + vowel_accuracy + consonant_accuracy
    n_total = len(data_train_loader)

    epoch_loss = total_loss / n_total
    epoch_accuracy = total_acc / (n_total * 3)

    # Calculate accuracy of each
    grapheme_accuracy = grapheme_accuracy / n_total
    vowel_accuracy = vowel_accuracy / n_total
    consonant_accuracy = consonant_accuracy / n_total

    return epoch_loss, epoch_accuracy, grapheme_accuracy, vowel_accuracy, consonant_accuracy
