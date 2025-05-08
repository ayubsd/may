import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Check input data
print("X:", X)
print("Y:", y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Predict
y_pred = mlr.predict(X_test)

# Evaluation
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Feature Importance
importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': mlr.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance:\n", importance)
