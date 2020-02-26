#!/bin/bash

source "/home/mod/.local/share/virtualenvs/Workspace-lCUZRoor/bin/activate"

for fold in 1 2 3;
do
    echo "fold ${fold}"
    python -m bengali.vowel_consonant.main \
          --test-size 64 \
          --base 'resnext50_32x4d' \
          --batch-size 64 \
          --fold ${fold} \
          --lr 3e-3 \
          --min-lr 1e-5 \
          --epochs 60 \
          --loss 'smooth' \
          --frozen False \
          --device 0 \
          --output-dir '/home/mod/Workspace/bengali-experiments/vc_64_resnext50_origin/'
done

for fold in 0 1 2 3;
do
    echo "fold ${fold}"
    python -m bengali.grapheme.main \
          --test-size 64 \
          --base 'resnext50_32x4d' \
          --batch-size 48 \
          --fold ${fold} \
          --lr 5e-3 \
          --min-lr 1e-5 \
          --max-height 10 \
          --max-width 10 \
          --loss 'smooth' \
          --frozen False \
          --cutmix-prob 0.6 \
          --no-cutmix False \
          --epochs 80 \
          --device 0 \
          --output-dir '/home/mod/Workspace/bengali-experiments/grapheme_64_resnext50_cutmix_origin/'
done