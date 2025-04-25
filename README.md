# Regression Models Comparison: Baseline vs. Lipschitz Constrained

## Overview

In this project, two regression models are implemented to predict housing prices using the California housing dataset:

1. **Baseline Regressor**: A simple regression model without any additional constraints or regularization.
2. **Lipschitz Regressor**: A regression model that incorporates Lipschitz regularization to control the model's sensitivity to input changes, ensuring smoother predictions.

The performance of both models is compared in terms of training loss, evaluation metrics, and model stability.

## Models

### 1. Baseline Regressor

The **Baseline Regressor** is a standard neural network model with no special constraints. It is trained to minimize the Mean Squared Error (MSE) between the predicted and true values.

### 2. Lipschitz Regressor

The **Lipschitz Regressor** is a variant of the baseline model, where Lipschitz regularization is applied. This regularization prevents the model from making large jumps in predictions, promoting stability and smoothness in the model's output.

## Training and Evaluation

### Data

The dataset used is the **California housing dataset**, which contains features about California districts and their respective median housing prices.

### Evaluation

The models were evaluated based on their **MSE** and **R²** scores on a held-out test set.
