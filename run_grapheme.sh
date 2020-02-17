#!/bin/bash

source "/home/mod961094/.local/share/virtualenvs/Workspace-n_bFWxY_/bin/activate"

for fold in 0 1 2 3;
do
    echo "fold ${fold}"
    python -m bengali.grapheme.main \
          --test-size 224 \
          --base 'resnext50_32x4d' \
          --batch-size 64 \
          --fold ${fold} \
          --lr 20e-3 \
          --epochs 90 \
          --device 1 \
          --output-dir '/home/mod961094/Workspace/bengali-experiments/'
done