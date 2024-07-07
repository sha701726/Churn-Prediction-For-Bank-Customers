# Churn Prediction for Bank Customers

# Objective
# The objective of this project is to predict customer churn for a bank using various machine learning techniques.
# Churn prediction helps in identifying customers who are likely to leave the bank, allowing the bank to take proactive measures to retain them.

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Import Data
df = pd.read_csv("bank.csv")
# print(df.head())
print(df.columns)
# Describe Data
print(df.info())
print(df.describe())

# Data Visualization
sns.countplot(x="Churn", data=df)
plt.show()

# Data Preprocessing
df = df.set_index("CustomerId")
df.replace({"Geography": {"France": 2, "Germany": 1, "Spain": 0}}, inplace=True)
df.replace({"Gender": {"Male": 2, "Female": 1}}, inplace=True)
df.replace({"NumOfProducts": {1: 0, 2: 1, 3: 1, 4: 1}}, inplace=True)
df["ZeroBalance"] = np.where(df["Balance"] > 0, 1, 0)

# Define Target Variable (y) and Feature Variables (X)
X = df.drop(["Surname", "Churn"], axis=1)
y = df["Churn"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2529)
x_rus, y_rus = RandomUnderSampler(random_state=2529).fit_resample(X, y)
x_ros, y_ros = RandomOverSampler(random_state=2529).fit_resample(X, y)
x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.8, random_state=2529)
x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.8, random_state=2529)

# Modeling
# Standardizing the features
sc = StandardScaler()
x_train[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]] = sc.fit_transform(x_train[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]])
x_test[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]] = sc.transform(x_test[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]])
x_train_rus[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]] = sc.fit_transform(x_train_rus[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]])
x_test_rus[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]] = sc.transform(x_test_rus[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]])
x_train_ros[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]] = sc.fit_transform(x_train_ros[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]])
x_test_ros[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]] = sc.transform(x_test_ros[["CreditScore", "Age", "Tenure", "Balance", "Estimated Salary"]])

# SVC Model
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

# Model Evaluation
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Hyperparameter Tuning
param_grid = {
    "C": [0.1, 1, 10],
    "gamma": [1, 0.1, 0.01],
    "kernel": ["rbf"],
    "class_weight": ["balanced"]
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=2)
grid.fit(x_train, y_train)
grid_predictions = grid.predict(x_test)

print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))

# Explanation
# This project aims to predict customer churn using the SVM model with hyperparameter tuning.
# Data preprocessing includes handling categorical data, creating new features, and standardizing the data.
# Random undersampling and oversampling techniques were also applied to handle class imbalance.
