# ==========================================
# IMPORT REQUIRED LIBRARIES
# ==========================================

import pandas as pd
# Used for data manipulation and analysis using DataFrames

import numpy as np
# Used for numerical computations

import matplotlib.pyplot as plt
# Used for plotting graphs and visualizations

import seaborn as sns
# Used for statistical data visualization

from sklearn.model_selection import train_test_split
# Used to split the dataset into training and testing sets

from sklearn.linear_model import LinearRegression
# Linear Regression model used for predicting continuous values

from sklearn.metrics import mean_squared_error, r2_score
# mean_squared_error: measures average squared prediction error
# r2_score: measures how well the model explains variance in the data

from sklearn.preprocessing import StandardScaler
# Used to standardize features so they are on a similar scale

from sklearn.datasets import fetch_california_housing
#



# ==========================================
# LOAD THE CALIFORNIA HOUSING DATASET
# ==========================================

housing = fetch_california_housing()
# Loads the California Housing dataset from scikit-learn



# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(housing.data, columns=housing.feature_names)
# Converts feature data into a pandas DataFrame

df['PRICE'] = housing.target
# Adds the target variable (median house price) as a new column



# ==========================================
# BASIC DATA EXPLORATION
# ==========================================

print("Dataset Shape:", df.shape)
# Prints number of rows and columns in the dataset

print(df.head())
# Displays the first 5 rows of the dataset



# ==========================================
# VISUALIZE HOUSE PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))
sns.histplot(df['PRICE'], bins=30, kde=True)
# Histogram showing distribution of house prices
# kde=True adds a smooth density curve

plt.title("Distribution of House Prices")
plt.show()



# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# Heatmap showing correlation between features
# Red indicates strong positive correlation
# Blue indicates negative or weak correlation

plt.title("Feature Correlation Heatmap")
plt.show()



# ==========================================
# DEFINE FEATURES AND TARGET
# ==========================================

X = df.drop('PRICE', axis=1)
# Independent variables (features)

y = df['PRICE']
# Dependent variable (target)



# ==========================================
# SPLIT DATA INTO TRAINING AND TEST SETS
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# 80% training data, 20% testing data
# random_state ensures reproducibility



# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()
# Initializes the scaler

X_train = scaler.fit_transform(X_train)
# Fits scaler on training data and scales it

X_test = scaler.transform(X_test)
# Scales test data using the same parameters



# ==========================================
# TRAIN LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()
# Initializes Linear Regression model

model.fit(X_train, y_train)
# Trains the model using training data



# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)
# Predicts house prices for test data



# ==========================================
# MODEL EVALUATION
# ==========================================

mse = mean_squared_error(y_test, y_pred)
# Mean Squared Error

rmse = np.sqrt(mse)
# Root Mean Squared Error

r2 = r2_score(y_test, y_pred)
# R-squared score

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")



# ==========================================
# ACTUAL VS PREDICTED COMPARISON
# ==========================================

comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
# Creates a table comparing actual and predicted prices

print(comparison.head())



# ==========================================
# VISUALIZE ACTUAL VS PREDICTED VALUES
# ==========================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
# Scatter plot comparing actual and predicted prices

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.show()
