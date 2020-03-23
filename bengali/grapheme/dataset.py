from typing import Callable, Tuple, List

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from bengali.utils.data_utils import get_image_path


def get_transform(
        *,
        train: bool,
        test_size: int,
        normalize: bool = True,
        no_cutmix: bool = True,
        max_height: int,
        max_width: int,
) -> Callable:

    if train and no_cutmix:
        transforms = [
            A.Resize(height=test_size, width=test_size),
            A.CoarseDropout(
                max_holes=8, max_height=max_height,
                max_width=max_width, fill_value=255, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            A.RandomBrightnessContrast(),
        ]
    elif train:
        transforms = [
            A.Resize(height=test_size, width=test_size),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            A.RandomBrightnessContrast(),
        ]
    else:
        transforms = [
            A.Resize(height=test_size, width=test_size),
        ]

    if normalize:
        transforms.append(A.Normalize())

    transforms.extend([
        ToTensorV2(),
    ])
    return A.Compose(transforms)


class GraphemeDataset(Dataset):
    def __init__(
            self, df: pd.DataFrame, root: str,
            transform: List[Callable],
    ):
        self.df = df
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple:
        item = self.df.iloc[idx]
        image = cv2.imread(get_image_path(item, self.root))

        transform = self.transform[idx % len(self.transform)]

        # get label
        if item.image_id.startswith('Train_'):
            label = item.grapheme_root
        else:
            label = 0

        data = {
            'image': image,
            'labels': label,
        }

        data = transform(**data)

        image = data['image']
        label = torch.tensor(data['labels'], dtype=torch.long)

        return image, label
