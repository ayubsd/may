import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import mode

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize lists to store WCSS (Within-Cluster Sum of Squares) and silhouette scores
wcss = []
silhouette_scores = []
cluster_range = range(2, 11)

# Find the optimal number of clusters using the elbow method and silhouette score
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

    if k > 1:
        silhouette_scores.append(silhouette_score(X_train, kmeans.labels_))
    else:
        silhouette_scores.append(0)

# Plot WCSS and silhouette scores
plt.figure(figsize=(12, 5))

# Elbow Method Plot
plt.subplot(1, 2, 1)
plt.plot(cluster_range, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# Silhouette Score Plot
plt.subplot(1, 2, 2)
plt.plot(cluster_range[1:], silhouette_scores[1:], marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Optimal number of clusters based on Elbow and Silhouette Method
optimal_k = 3

# Fit KMeans model with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans.fit(X_train)

# Predict clusters on the test data
test_clusters = kmeans.predict(X_test)

# Assign cluster labels to the true labels based on the mode (most frequent) for each cluster
labels = np.zeros_like(test_clusters)
for i in range(optimal_k):
    mask = (test_clusters == i)
    labels[mask] = mode(y_test[mask])[0]

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, labels))
print("\nClassification Report:")
print(classification_report(y_test, labels))

# Plot clustering results on test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_clusters, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Centroids')
plt.title('K-means Clustering on Iris Dataset (Test Data)')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.show()

# Display cluster centers in the original scale
print("\nCluster Centers (in original scale):")
original_scale_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print(original_scale_centers)
