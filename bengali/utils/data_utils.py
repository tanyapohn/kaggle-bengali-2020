import os
from typing import Tuple

import pandas as pd

ROOT = os.path.dirname('/home/mod/Workspace/kaggle/Bengali/')
# DATA_ROOT = os.path.join(ROOT, 'grapheme-imgs-origin')
DATA_ROOT = os.path.join(ROOT, 'grapheme-imgs-128x128')


def get_image_path(item: pd.DataFrame, root: str) -> str:
    """ Train and test are in the same folder """
    path = os.path.join(root, f'{item.image_id}.png')
    return path


def load_train_df(root: str = ROOT, train=True) -> pd.DataFrame:
    if train:
        df_path = os.path.join(root, 'train_with_fold.csv')
    else:
        df_path = os.path.join(root, 'test.csv')
    return pd.read_csv(df_path)


def load_train_valid_df(fold: int, task: str) -> Tuple:

    df = load_train_df()
    if task == 'grapheme':
        classes = int((df.nunique())[1:2])
    else:
        # vowel and consonant
        classes = list(df.nunique())[2:-3]

    mask = df['fold'] == fold
    train = df[~mask]
    valid = df[mask]
    return train, valid, classes



