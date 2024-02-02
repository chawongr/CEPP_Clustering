import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Function to plot data and clusters
def plot_clusters(X, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

# Generate data
X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=42)

# Use KMeans for clustering
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)

# Calculate ARI
ari_score = adjusted_rand_score(y, labels)
print(f"Adjusted Rand Index (ARI): {ari_score}")

# Plot the results
plt.figure(figsize=(12, 6))

# Original data and clustered data
plt.subplot(1, 2, 1)
plot_clusters(X, y, title='Original Data')

plt.subplot(1, 2, 2)
plot_clusters(X, labels, title=f'KMeans Clustering (ARI={ari_score:.2f})')

plt.tight_layout()
plt.show()
