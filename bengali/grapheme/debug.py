import argparse
import os
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader

from bengali.grapheme.dataset import get_transform, GraphemeDataset
from bengali.grapheme.evaluator import eval_one_epoch
from bengali.grapheme.models import build_model
from bengali.grapheme.trainer import train_one_epoch
from bengali.utils.data_utils import DATA_ROOT, ROOT


def save_checkpoint(
        state: Dict, output_dir: str, fold: int,
        is_best: bool, best_acc: float,
):
    torch.save(state, os.path.join(output_dir, f'checkpoint_{fold}.pth'))
    if is_best:
        # shutil.copyfile(filename, 'model_best.pth')
        print(f'Update best model with accuracy: {best_acc}\n')
        torch.save(state['state_dict'], os.path.join(output_dir, f'model_best_{fold}.pth'))


def main():
    """
    Simple model trainer
    """
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--test-size', type=int, default=224)
    arg('--base', default='resnet50')

    # Training params
    arg('--batch-size', type=int, default=64)
    arg('--fold', type=int)
    arg('--lr', default=25e-3, type=float, help='initial learning rate')
    arg('--min-lr', default=1e-10, type=float)
    arg('--optimizer', default='sgd', type=str)
    arg('--momentum', default=0.9, type=float)
    arg('--weight-decay', default=1e-4, type=float, help='weight decay')
    arg('--scheduler', default='cosine', type=str)
    arg('--epochs', type=int)
    arg('--fp16', default=True, type=bool)
    arg('--pretrained', default=True, type=bool)
    arg('--cutmix-prob', default=0.5, type=float)
    arg('--device', default=0, type=int)

    # Save path param
    arg('--output-dir', type=str)

    args = parser.parse_args()

    def _get_transforms(*, normalise: bool = True, train: bool):
        test_sizes = [args.test_size]
        return [
            get_transform(
                train=train,
                test_size=test_size,
                normalize=normalise,
            ) for test_size in test_sizes]

    def make_test_data_loader(df):
        return DataLoader(
            GraphemeDataset(
                df=df,
                transform=_get_transforms(train=False),
                root=DATA_ROOT,
            ),
            batch_size=args.batch_size,
        )

    def load_train_df(root: str = ROOT, train=True) -> pd.DataFrame:
        if train:
            df_path = os.path.join(root, 'train_debug.csv')
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

    train_df, valid_df, classes = load_train_valid_df(args.fold, task='grapheme')
    data_test_loader = make_test_data_loader(valid_df)
    data_train_loader = DataLoader(
        GraphemeDataset(
            df=train_df,
            transform=_get_transforms(train=True),
            root=DATA_ROOT,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    model: Module = build_model(
        base=args.base,
        pretrained=args.pretrained,
        n_classes=classes,
    )
    torch.backends.cudnn.benchmark = True
    print('Creating model ...')
    model = model.to(device)
    parameters = model.parameters()

    print(model)

    loss = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            params=parameters, lr=args.lr,
            weight_decay=args.weight_decay, momentum=args.momentum,
        )
    else:
        optimizer = optim.Adam(params=parameters, lr=args.lr)

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr,
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', verbose=True,
            factor=0.77, patience=2, min_lr=args.min_lr,
        )

    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O1",
        )

    log = {
        'epoch': [],
        'lr': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_acc': [],
    }
    best_acc = 0
    for epoch in range(args.epochs):

        curr_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print(f'Current learning rate: {curr_lr}')
        # train
        train_loss, train_acc = train_one_epoch(
            data_train_loader, model, args.cutmix_prob,
            args.fp16, device, loss, optimizer,
        )
        # eval
        valid_loss, valid_acc = eval_one_epoch(
            data_test_loader, model,
            device, loss,
        )

        # update scheduler
        if args.scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(valid_loss)

        print(f'\nTrain loss: {train_loss}\t Train acc: {train_acc}')

        print(f'Valid loss: {valid_loss}\t Valid acc: {valid_acc}')

        # save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'lr_scheduler': scheduler.state_dict(),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, args.output_dir, args.fold, is_best, best_acc)

        log['epoch'].append(epoch + 1)
        log['lr'].append(curr_lr)
        log['train_loss'].append(train_loss)
        log['train_acc'].append(train_acc)
        log['val_loss'].append(valid_loss)
        log['val_acc'].append(valid_acc)
        log['best_acc'].append(best_acc)

        # writing log
        pd.DataFrame(log).to_csv(os.path.join(args.output_dir, f'log_grapheme_{args.fold}.csv'), index=False)


if __name__ == '__main__':
    main()
