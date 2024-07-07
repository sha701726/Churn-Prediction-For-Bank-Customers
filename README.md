# Churn Prediction for Bank Customers

## Objective

The objective of this project is to predict customer churn for a bank using various machine learning techniques. Churn prediction helps in identifying customers who are likely to leave the bank, allowing the bank to take proactive measures to retain them.

## Data Source

The dataset used for this project is bank.csv, which contains information about bank customers. This dataset includes features such as customer ID, age, geography, gender, balance, and whether or not the customer churned.

## Import Library

- pandas as pd
- numpy as np
- matplotlib.pyplot as plt
- seaborn as sns
- sklearn.model_selection as train_test_split
- sklearn.preprocessing as StandardScaler
- sklearn.svm as SVC
- sklearn.metrics as confusion_matrix, accuracy_score, classification_report
- imblearn.under_sampling as RandomUnderSampler
- imblearn.over_sampling as RandomOverSampler

## Import Data

Load the dataset from bank.csv.

## Describe Data

Display data information and summary statistics using `df.info()` and `df.describe()`.

## Data Visualization

Visualize the distribution of churned customers using `sns.countplot()`.

## Data Preprocessing

Handle categorical data, create new features, and standardize the data using `StandardScaler`.

## Define Target Variable (y) and Feature Variables (X)

Define the target variable (churn) and feature variables (all other columns).

## Train Test Split

Split data into training and testing sets using `train_test_split`.

## Modeling

Create an SVM model and standardize features.

## Model Evaluation

Evaluate the model's performance using `confusion_matrix`, `accuracy_score`, and `classification_report`.

## Prediction

Make predictions using the model.

## Hyperparameter Tuning

Perform hyperparameter tuning using GridSearchCV.

## Explanation

This project aims to predict customer churn using the SVM model with hyperparameter tuning. Data preprocessing includes handling categorical data, creating new features, and standardizing the data. Random undersampling and oversampling techniques were also applied to handle class imbalance.
