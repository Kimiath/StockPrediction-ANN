# NASDAQ Stock Price Prediction using Artificial Neural Networks (ANN)

This project explores a complete machine learning workflow for predicting NASDAQ stock prices using **Artificial Neural Networks (ANNs)**.  
It combines conceptual learning of neural networks across multiple frameworks with a final end-to-end ANN implementation that includes **feature engineering, training, overfitting control, and evaluation**.

## Project Overview

The project is structured in two parts:

### 1. Neural Network Learning & Exploration
To build a strong understanding of neural networks, I implemented ANNs using several approaches and frameworks:
- Manual neural network implementation (from scratch)
- ANN using scikit-learn
- ANN using TensorFlow/Keras
- ANN using PyTorch
- Experiments with **dropout** and **early stopping** techniques to prevent model overfitting

These learning experiments are stored in the `experiments/` directory.

### 2. End-to-End ANN Stock Prediction
After gaining familiarity with ANN concepts and frameworks, I built a complete stock price prediction pipeline using a single Python file (`main.py`), which includes:
- Data loading and preprocessing
- Feature engineering
- Model definition and training
- Overfitting control
- Model evaluation and visualization

To reflect a realistic data science workflow for financial prediction tasks.

## Main Model Pipeline (`main.py`)

The `main.py` file contains the **full pipeline**, including:
- Feature engineering and scaling
- Train/validation split
- ANN model training
- Overfitting control techniques (dropout, early stopping)
- Performance evaluation (loss curves, error metrics)
- Prediction vs. actual price visualization

## `Datasets
- The Dataset folder includes two folders:
- Cal-housing Dataset is used for experimential learning and comparing different neural network models(with different frameoworks) on the same dataset to evaluate the best outcome.
- StockData is the dataset used for the main project's model training. 
