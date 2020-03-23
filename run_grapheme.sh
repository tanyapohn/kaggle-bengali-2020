#!/bin/bash

# to activate the virtual environment, change here
source "/home/mod/.local/share/virtualenvs/Workspace-lCUZRoor/bin/activate"

# Grapheme with cutmix
for fold in 0 1 2 3;
do
    echo "fold ${fold}"
    python -m bengali.grapheme.main \
          --test-size 128 \
          --base 'se_resnext50_32x4d' \
          --batch-size 48 \
          --fold ${fold} \
          --lr 3e-3 \
          --min-lr 1e-5 \
          --loss 'smooth' \
          --cutmix-prob 0.5 \
          --epochs 45 \
          --device 0 \
          --output-dir '/home/mod/Workspace/bengali-experiments/grapheme_64_se_resnext101_cutmix/'
done