#!/bin/bash

source "/home/mod/.local/share/virtualenvs/Workspace-lCUZRoor/bin/activate"

for fold in 0 1 2 3;
do
    echo "fold ${fold}"
    python -m bengali.vowel_consonant.main \
          --test-size 224 \
          --base 'densenet169' \
          --batch-size 64 \
          --fold ${fold} \
          --lr 11e-3 \
          --min-lr 1e-5 \
          --epochs 45 \
          --device 0 \
          --output-dir '/home/mod/Workspace/vowel_consonant_models/'
done