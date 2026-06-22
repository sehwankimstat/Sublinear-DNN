Sublinearly Structured Deep Neural Networks Achieve Feature Learning Consistency for Compositional Functions
===============

## Related Publication

Sehwan Kim, Yan Sun, Faming Liang (2026+), [Sublinearly Structured Deep Neural Networks Achieve Feature Learning Consistency for Compositional Functions](..), accepted to *Statistical Learning and Data Science*, Special Issue on Statistics and AI.

## Overview

This repository contains the code used in the paper *Sublinearly Structured Deep Neural Networks Achieve Feature Learning Consistency for Compositional Functions*.

The paper studies feature learning consistency of deep neural networks for compositional functions. In particular, it investigates whether sublinearly structured DNNs can recover the underlying feature structure of the target function.

## Code Description

* `cca_calculation.py`:
  Evaluates whether sublinear DNNs recover the true feature structure in synthetic regression experiments. The script computes canonical correlations between the true and learned feature eigenspaces, and reports training/test MSE for different network structures.

* `MNIST_double_descent/`:
  Contains MNIST experiments for studying the double descent phenomenon across different neural network widths. This folder is adapted from the original PyTorch implementation by Songwei Ge:
  [SongweiGe/double-descent-pytorch](https://github.com/SongweiGe/double-descent-pytorch).
  We modified the code for our experimental setting and analysis.

* `Narrow_DNN_CelebA_mlp.ipynb`:
  Trains a fully connected neural network on a balanced binary CelebA attribute classification task. The notebook constructs a custom CelebA dataset, selects a balanced subset based on a chosen attribute such as `Eyeglasses`, trains an MLP classifier, and visualizes the dominant first-layer feature direction through the top eigenvector of $W^\top W$.
