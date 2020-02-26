import os
from typing import Dict, Tuple

import pandas as pd
import torch

ROOT = os.path.dirname('/home/mod/Workspace/kaggle/Bengali/')
# DATA_ROOT = os.path.join(ROOT, 'grapheme-imgs-origin')
DATA_ROOT = os.path.join(ROOT, 'grapheme-imgs-137x236')


def save_checkpoint(
        state: Dict, output_dir: str, fold: int,
        is_best: bool, best_acc: float,
):
    torch.save(state, os.path.join(output_dir, f'checkpoint_{fold}.pth'))
    if is_best:
        # shutil.copyfile(filename, 'model_best.pth')
        print(f'Update best model with accuracy: {best_acc}\n')
        torch.save(state['state_dict'], os.path.join(output_dir, f'model_best_{fold}.pth'))


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
    elif task == 'vowel+consonant':
        # vowel and consonant
        classes = list(df.nunique())[2:-3]
    else:
        classes = list(df.nunique())[1:-3]

    mask = df['fold'] == fold
    train = df[~mask]
    valid = df[mask]
    return train, valid, classes



