from typing import Callable, Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval_one_epoch(
        data_test_loader: DataLoader,
        model: Module, device, loss: Callable,
) -> Tuple:

    total_loss = 0
    grapheme_accuracy = 0
    vowel_accuracy = 0
    consonant_accuracy = 0

    model.eval()
    with torch.no_grad():
        for idx, (input_image, labels) in tqdm(
                enumerate(data_test_loader), total=len(data_test_loader),
        ):
            input_image = input_image.to(device, dtype=torch.float)
            grapheme_label = labels[:, 0].to(device, dtype=torch.long)
            vowel_label = labels[:, 1].to(device, dtype=torch.long)
            consonant_label = labels[:, 2].to(device, dtype=torch.long)

            pred_grapheme, pred_vowel, pred_consonant = model(input_image)

            loss_grapheme = loss(pred_grapheme, grapheme_label)
            loss_vowel = loss(pred_vowel, vowel_label)
            loss_consonant = loss(pred_consonant, consonant_label)

            losses = 0.5*loss_grapheme + 0.25*loss_vowel + 0.25*loss_consonant

            total_loss += losses.item()

            grapheme_accuracy += (pred_grapheme.argmax(1) == grapheme_label).float().mean().item()
            vowel_accuracy += (pred_vowel.argmax(1) == vowel_label).float().mean().item()
            consonant_accuracy += (pred_consonant.argmax(1) == consonant_label).float().mean().item()

        total_acc = grapheme_accuracy + vowel_accuracy + consonant_accuracy
        n_total = len(data_test_loader)

        epoch_loss = total_loss / n_total
        epoch_accuracy = total_acc / (n_total * 3)

        # Calculate accuracy of each
        grapheme_accuracy = grapheme_accuracy / n_total
        vowel_accuracy = vowel_accuracy / n_total
        consonant_accuracy = consonant_accuracy / n_total

    return epoch_loss, epoch_accuracy, grapheme_accuracy, vowel_accuracy, consonant_accuracy
