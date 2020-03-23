#!/bin/bash

# to activate the virtual environment, change here
source "/home/mod/.local/share/virtualenvs/Workspace-lCUZRoor/bin/activate"


python -m bengali.vowel_consonant.main \
      --test-size 128 \
      --base 'resnext50_32x4d' \
      --batch-size 64 \
      --fold 2 \
      --lr 3e-3 \
      --min-lr 1e-5 \
      --epochs 60 \
      --loss 'cross_entropy' \
      --frozen False \
      --device 0 \
      --output-dir '/home/mod/Workspace/bengali-experiments/vc_64_resnext50_origin/'


# Grapheme no cutmix
python -m bengali.grapheme.main \
      --test-size 128 \
      --base 'resnext50_32x4d' \
      --batch-size 48 \
      --fold 2 \
      --lr 5e-3 \
      --min-lr 1e-5 \
      --max-height 10 \
      --max-width 10 \
      --loss 'cross_entropy' \
      --frozen False \
      --cutmix-prob 0.6 \
      --no-cutmix False \
      --epochs 80 \
      --device 0 \
      --output-dir '/home/mod/Workspace/bengali-experiments/grapheme_64_resnext50_cutmix_origin/'
