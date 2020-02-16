from typing import Callable, Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval_one_epoch(
        data_test_loader: DataLoader, model: Module,
        device, loss: Callable,
) -> Tuple:

    cum_grapheme_loss = 0
    cum_accuracy = 0

    model.eval()
    with torch.no_grad():
        for idx, (input_image, label) in tqdm(
                enumerate(data_test_loader), total=len(data_test_loader),
        ):
            input_image = input_image.to(device, dtype=torch.float)
            grapheme_label = label.to(device, dtype=torch.long)
            pred_grapheme = model(input_image)

            grapheme_loss = loss(pred_grapheme, grapheme_label)

            cum_grapheme_loss += grapheme_loss.item()
            cum_accuracy += (pred_grapheme.argmax(1) == grapheme_label).float().mean().item()

        n_total = len(data_test_loader)
        epoch_loss = cum_grapheme_loss / n_total
        epoch_acc = cum_accuracy / n_total

    return epoch_loss, epoch_acc
