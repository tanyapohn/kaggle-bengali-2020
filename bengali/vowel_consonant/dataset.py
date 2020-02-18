from typing import Callable, Tuple
from typing import List

import albumentations as A
import cv2
import numpy as np
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
        ) -> Callable:
    if train:
        transforms = [
            A.CoarseDropout(
                max_holes=8, max_height=40,
                max_width=40, p=0.6),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=30,
                p=0.5
            ),
        ]
    else:
        transforms = [
            A.LongestMaxSize(test_size),
        ]

    if normalize:
        transforms.append(A.Normalize(mean=(0.0692, 0.0692, 0.0692),
                                      std=(0.2052, 0.2052, 0.2052)))

    transforms.extend([
        ToTensorV2(),
    ])
    return A.Compose(transforms)


class VowelConsonantDataset(Dataset):
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

        # get labels
        if item.image_id.startswith('Train_'):
            labels = [
                item.vowel_diacritic,
                item.consonant_diacritic,
            ]
        else:
            labels = np.zeros((0, 2))

        data = {
            'image': image,
            'labels': labels,
        }

        data = transform(**data)

        image = data['image']
        labels = torch.tensor(data['labels'], dtype=torch.long)

        return image, labels
