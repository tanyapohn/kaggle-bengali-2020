#!/bin/bash

source "/home/mod961094/.local/share/virtualenvs/Workspace-n_bFWxY_/bin/activate"

for fold in 0 1 2 3;
do
    echo "fold ${fold}"
    python -m bengali.vowel_consonant.debug \
          --test-size 224 \
          --base 'densenet169' \
          --batch-size 64 \
          --fold ${fold} \
          --lr 15e-3 \
          --epochs 5 \
          --device 0 \
          --output-dir '/home/mod961094/Workspace/vowel-consonants-experiments/'
done