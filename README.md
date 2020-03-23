# Kaggle-Bengali-2020

74th place solution with a private score of `0.9361` for 
[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)

Contents
 - [Setup](#setup)
 - [Overview](#overview)
 - [Solutions](#solutions)
 - [Training](#training)
 - [Submission](#submission)

## Setup

Python 3.6+

To install the virtual environment:
```
$ pipenv install -r requirements.txt --python 3.6
```
or 
```
$ pip install -r requirements.txt
```

**Optional:** To install apex, please check this [github](https://github.com/NVIDIA/apex).

## Overview

This repo contains many different approaches of implementing classification models for
Bengali. For each approach, it's been briefed in this following table:

| Experiments                   | Backbone                  | LB Score | Private Score |
| ----------------------------- | -------------             | :------: | :-----------: |
| Grapheme                      | ResNext50                  |          |               |
| Vowel + Consonant             | STN+ResNext50             | 0.9665   | **0.9361**    |
| Grapheme + Vowel + Consonant  | SeResNext50               | 0.9776   | 0.9242        |
| Grapheme + Vowel + Consonant  | SeResNext50 + DenseNet169 | 0.9770   | 0.9261        |

The highest private score is the experiment of separated training between `Grapheme` and
`Vowel + Consonant`.

`bengali/grapheme`: is grapheme model\
`bengali/vowel_consonant`: is vowel and grapheme model with 
[`Spatial Transformer Network`](https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/transformation.py)\
`bengali/grapheme_vowel_consonant`: is the model with 3 heads

The reason behind training them separately is to avoid overfitting of `Vowel` and
`Consonant`, also to tweak the accuracy of `Grapheme` independently.

## Solutions

#### Preprocessing

Use this [public kernel](https://www.kaggle.com/iafoss/image-preprocessing-128x128) to
process the images size of 224x224

#### Augmentation
For `Spatial Transformer Network`, I didn't do any extra augmentations.
```python
import albumentations as A

# for other experiments
# A.CoarseDropout or Cutmix
# A.ShiftScaleRotate
# A.RandomBrightnessContrast
A.Compose(
    A.Normalize(mean=(0.0692, 0.0692, 0.0692),
                std=(0.2052, 0.2052, 0.2052)),
    ToTensorV2(),
)
```

#### Grapheme Model
- Backbone: Resnet50
- Loss: CrossEntropy
- Optimizer: SGD
- Scheduler: Cosine Annealing
- Epochs: 80 if `Cutmix` else 40-60 

#### Vowel+Consonant Model
- Backbone: STN+ResNext50
- Loss: CrossEntropy
- Optimizer: SGD
- Scheduler: Cosine Annealing
- Epochs: 80-100

## Training

Before training a model, you might need to customise the `arguments` in `run_grapheme.sh`
and `run_vowel_consonant.sh` to fit with your system.

To train models:
```
$ bash /path/to/project/run_grapheme.sh
$ bash /path/to/project/run_vowel_consonant.sh
```

## Submission
Code for submission has been written in this [kernel](https://www.kaggle.com/moximo13/spatial-transform-network-bengali?scriptVersionId=30658074)