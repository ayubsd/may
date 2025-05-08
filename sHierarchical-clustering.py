import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a dendrogram to visualize the hierarchical clustering structure
plt.figure(figsize=(12, 6))
plt.title('Dendrogram for Iris Dataset (Ward Linkage)')
dend = dendrogram(linkage(X_train, method='ward'))
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

# Define the optimal number of clusters
optimal_clusters = 3

# Perform Agglomerative Clustering on the training data
agg_clustering = AgglomerativeClustering(n_clusters=optimal_clusters, metric='euclidean', linkage='ward')
train_clusters = agg_clustering.fit_predict(X_train)

# Predict clusters on the test data
test_clusters = agg_clustering.fit_predict(X_test)

# Assign labels based on mode for each cluster in the test data
labels = np.zeros_like(test_clusters)
for i in range(optimal_clusters):
    mask = (test_clusters == i)
    labels[mask] = mode(y_test[mask])[0]

# Print evaluation metrics for the test data
print("\nEvaluation Metrics for Test Data:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, labels))
print("\nClassification Report:")
print(classification_report(y_test, labels))

# Calculate the silhouette score for the clustering
silhouette_avg = silhouette_score(X_test, test_clusters)
print(f"\nSilhouette Score: {silhouette_avg:.3f}")

# Perform PCA to reduce the data to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

# Visualize the clustering results using the first two principal components
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=test_clusters, cmap='viridis', s=50)
plt.title('Hierarchical Clustering on Iris Dataset (Test Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Print the feature importance in the principal components
print("\nFeature Importance in Principal Components:")
pca_components = pd.DataFrame(pca.components_, columns=feature_names, index=['PC1', 'PC2'])
print(pca_components)

# Visualize the hierarchical clustering using only the sepal features
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_clusters, cmap='viridis', s=50)
plt.title('Hierarchical Clustering (Using Sepal Features)')
plt.xlabel(feature_names[0])  # Sepal Length
plt.ylabel(feature_names[1])  # Sepal Width
plt.colorbar(label='Cluster')
plt.show()
