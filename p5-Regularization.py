# Mount Drive and import libraries
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess data
data = pd.read_csv('/content/drive/MyDrive/ML/housing_FULL.csv')

# Select relevant columns
columns = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount',
           'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
data = data[columns]

# Fill missing values
columns_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
data[columns_to_fill_zero] = data[columns_to_fill_zero].fillna(0)
data['Landsize'] = data['Landsize'].fillna(data['Landsize'].mean())
data['BuildingArea'] = data['BuildingArea'].fillna(data['BuildingArea'].mean())

# Drop rows where 'Price', 'Regionname', or 'CouncilArea' are still null
data.dropna(inplace=True)

# Separate features and target
x = data.drop('Price', axis=1)
y = data['Price']

# One-hot encode categorical variables
x = pd.get_dummies(x, columns=['Suburb', 'Type', 'Method', 'SellerG', 'Regionname', 'CouncilArea'])

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(x_train, y_train)
lasso_preds = lasso.predict(x_test)

# Train Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
ridge_preds = ridge.predict(x_test)

# Evaluate both models
print("Lasso Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, lasso_preds)))
print("R2 Score:", r2_score(y_test, lasso_preds))

print("\nRidge Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, ridge_preds)))
print("R2 Score:", r2_score(y_test, ridge_preds))

# Plot predictions vs actuals for both
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lasso_preds, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso Regression")

plt.subplot(1, 2, 2)
plt.scatter(y_test, ridge_preds, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Ridge Regression")

plt.tight_layout()
plt.show()
