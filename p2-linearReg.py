import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load California Housing dataset
data = fetch_california_housing()
X = data.data[:, [0]]  # Use only 1 feature for simple linear regression
y = data.target

# Check data
print("X:", X)
print("Y:", y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Evaluate
print("Mean squared error:", mean_squared_error(y_test, y_pred))
print("R2 score:", r2_score(y_test, y_pred))

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test.ravel(), y=y_test, alpha=0.7, color='blue', label='Actual Data')
sns.regplot(x=X_test.ravel(), y=y_pred, scatter=False, color='red', line_kws={'linewidth': 2}, label='Regression Line')
plt.xlabel("Feature (MedInc)", fontsize=12)
plt.ylabel("Target (House Value)", fontsize=12)
plt.title("Simple Linear Regression", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
