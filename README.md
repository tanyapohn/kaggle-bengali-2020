# Kaggle-Bengali-2020

74th place solution with a private score of `0.9361` for 
[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19)

Contents
 - [Setup](#setup)
 - [Overview](#overview)
 - Solutions
 - Training
 - Submission

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
| ----------------------------- | -------------             | -------- | ------------- |
| Grapheme                      | Resnet50                  |          |               |
| Vowel + Consonant             | STN+ResNext50             | 0.9665   | **0.9361**    |
| Grapheme + Vowel + Consonant  | SeResNext50               | 0.9776   | 0.9242        |
| Grapheme + Vowel + Consonant  | SeResNext50 + DenseNet169 | 0.9770   | 0.9261        |