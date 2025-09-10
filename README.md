# Churn-Modeling-ANN

This repository contains an implementation of an Artificial Neural Network (ANN) for predicting customer churn using Python and TensorFlow/Keras. The model is trained on the popular "Churn_Modelling.csv" dataset, which contains data from a bank’s customers to predict whether a customer will exit (churn) or not.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

---

## Overview

The goal of this project is to build an ANN that accurately predicts customer churn based on various features such as credit score, geography, gender, age, tenure, balance, and more. The notebook guides you through data preprocessing, feature scaling, model building, training, and evaluation.

## Dataset

- **Input:** `Churn_Modelling.csv`
    - Contains customer data including demographic and account information.
    - The target variable is `Exited` (1 if the customer left, 0 otherwise).

## Features Used

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumberOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

## Project Workflow

1. **Data Preprocessing:**
    - Importing data using `pandas`
    - Selecting relevant features and target variable
    - Handling categorical data (e.g., Geography, Gender)
    - Splitting the dataset into training and test sets
    - Feature scaling using `StandardScaler`

2. **Model Building:**
    - Creating an ANN using TensorFlow's Keras Sequential API
    - Adding input, hidden, and output layers
    - Using the Adam optimizer with a custom learning rate

3. **Training:**
    - Early stopping callback to avoid overfitting
    - Training with validation split

4. **Evaluation:**
    - Making predictions on the test set
    - Evaluating with a confusion matrix and accuracy score
    - Plotting training and validation loss curves

## Model Architecture

- **Input Layer:** 11 features (after encoding)
- **Hidden Layers:** Dense layers with activation functions (e.g., ReLU)
- **Output Layer:** 1 neuron with sigmoid activation (binary classification)
- **Optimizer:** Adam with custom learning rate
- **Loss Function:** Binary Crossentropy

## Evaluation

- **Confusion Matrix** and **Accuracy Score** are used to assess the model's performance.
- Example output:
    ```
    Confusion Matrix:
    [[1546   49]
     [ 230  175]]

    Accuracy Score: ~0.86
    ```
- **Loss Curves**: The model's loss over epochs is plotted for both training and validation sets.

## Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/ranjeet22/Churn-Modeling-ANN.git
    cd Churn-Modeling-ANN
    ```

2. Ensure you have the required libraries installed (see below).

3. Place the `Churn_Modelling.csv` file in the same directory as the notebook.

4. Open and run the `ANN CODE.ipynb` notebook in Jupyter Notebook or Google Colab.

## Requirements

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- pandas
- numpy
- matplotlib

Install requirements with:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

## Acknowledgements

- Dataset source: [Kaggle - Churn Modelling](https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers)
---

Feel free to open issues or submit pull requests for improvements or questions!
