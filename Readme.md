Sublinearly Structured Deep Neural Networks Achieve Feature Learning Consistency for Compositional Functions
===============

## Related Publication

Sehwan Kim, Yan Sun, Faming Liang (2026+), [Sublinearly Structured Deep Neural Networks Achieve Feature Learning Consistency for Compositional Functions](..), accepted to *Statistical Learning and Data Science*, Special Issue on Statistics and AI.

## Overview

This repository contains the code used in the paper. 

- `cca_calculation.py`: Evaluates whether sublinear DNNs recover the true feature structure in synthetic regression experiments. The script computes canonical correlations between the true and learned feature eigenspaces, and reports training/test MSE for different network structures.

- `Narrow_DNN_CelebA_mlp.ipynb`: Trains a fully connected neural network on a balanced binary CelebA attribute classification task. The notebook constructs a custom CelebA dataset, selects a balanced subset based on a chosen attribute such as `Eyeglasses`, trains an MLP classifier, and visualizes the dominant first-layer feature direction through the top eigenvector of $W^\top W$.


## Results

The notebook `DoubleNN_nonlinear_nonlinear.ipynb`.describes the application of Double NN to the nonlinear $c(\cdot)$ and nonlinear $\tau(\cdot)$. Please refer Section 5 at [Extended Fiducial Inference for Individual Treatment Effects via Deep Neural Networks](https://arxiv.org/abs/2505.01995) for further detail.

<p align="center">
    <img src="DoubleNN_example.png" width=600>
</p>

The above figure presents the results of Double-NN: (left) a scatter plot of $\hat{z}_{i}$ (y-axis) versus $z_i$ (x-axis), (middle-left) a Q-Q plot of $\hat{z}_i$ and $z_i$, (middle-right) scatter plot of $\tau(x)$ and $\hat{\tau}(x)$ , and (right) scatter plot of $c(x)$ and $\hat{c}(x)$

The left panel shows that the imputed random errors closely match the true random errors. The middle-left panel demonstrates that the imputed errors exhibit a distributional behavior similar to that of the true errors. The middle-right and right panels indicate that neural networks trained using the Double-NN method can accurately estimate the true response surface. By collecting these values across iterations, one can construct confidence or prediction intervals for $c(\cdot)$ and $\tau(\cdot)$. 



