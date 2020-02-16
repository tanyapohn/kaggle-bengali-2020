#!/bin/bash

source "/home/mod/.local/share/virtualenvs/Workspace-lCUZRoor/bin/activate"

python -m bengali.grapheme.debug \
      --test-size 128 \
      --base 'resnet50' \
      --batch-size 64 \
      --fold 0 \
      --lr 11e-3 \
      --epochs 30 \
      --output-dir '/home/mod/Workspace/debug/'
