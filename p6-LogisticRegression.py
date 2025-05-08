# Importing necessary libraries
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    roc_curve,
    roc_auc_score
)

# Load and prepare the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Use only two classes for binary classification
df = df[df['target'] != 2]

# Define features and target
x = df.drop('target', axis=1)
y = df['target']

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# Train logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Accuracy and error
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Error:", 1 - accuracy)

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
print("AUC:", auc)

plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 2)))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=10)
cv_mean_accuracy = cross_val_score(model, x, y, cv=kfold, scoring='accuracy').mean()
print("Cross-validation mean accuracy:", cv_mean_accuracy)
