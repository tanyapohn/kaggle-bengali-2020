from typing import Callable, Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
        data_train_loader: DataLoader, model: Module,
        fp16: bool, device, loss: Callable, optimizer,
) -> Tuple:

    total_loss = 0
    vowel_accuracy = 0
    consonant_accuracy = 0

    model.train()
    for idx, (input_image, labels) in tqdm(
            enumerate(data_train_loader), total=len(data_train_loader),
    ):
        input_image = input_image.to(device, dtype=torch.float)
        vowel_label = labels[:, 0].to(device, dtype=torch.long)
        consonant_label = labels[:, 1].to(device, dtype=torch.long)

        pred_vowel, pred_consonant = model(input_image)

        loss_vowel = loss(pred_vowel, vowel_label)
        loss_consonant = loss(pred_consonant, consonant_label)

        losses = loss_vowel + loss_consonant

        optimizer.zero_grad()

        if fp16:
            from apex import amp
            with amp.scale_loss(losses, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses.backward()

        optimizer.step()

        total_loss += losses.item()

        vowel_accuracy += (pred_vowel.argmax(1) == vowel_label).float().mean().item()
        consonant_accuracy += (pred_consonant.argmax(1) == consonant_label).float().mean().item()

    total_acc = vowel_accuracy + consonant_accuracy
    n_total = len(data_train_loader)

    epoch_loss = total_loss / n_total
    epoch_accuracy = total_acc / (n_total * 2)

    # Calculate accuracy of each
    vowel_accuracy = vowel_accuracy / n_total
    consonant_accuracy = consonant_accuracy / n_total

    return epoch_loss, epoch_accuracy, vowel_accuracy, consonant_accuracy
