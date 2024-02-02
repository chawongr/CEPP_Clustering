import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.metrics import silhouette_score

# Generate complex blobs data
X, _ = make_blobs(n_samples=500, cluster_std=5.0)

class CustomKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids_history = []

    def fit(self, X):
        # Randomly initialize centroids
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            # Assign each data point to the nearest centroid
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)

            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Calculate movement of centroids
            centroid_movement = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.centroids_history.append(self.centroids.copy())  # Record centroid movements

            if centroid_movement < self.tol:
                print(f"Converged after {i + 1} iterations.")
                break

    def silhouette_score(self, X):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)
        return silhouette_score(X, labels)

    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)

# Fit the custom KMeans model for different numbers of clusters and compute silhouette scores
def find_best_n_clusters(X, min_clusters, max_clusters):
    silhouette_scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        custom_kmeans = CustomKMeans(n_clusters=n_clusters)
        custom_kmeans.fit(X)
        silhouette_scores.append(custom_kmeans.silhouette_score(X))
    best_num_clusters = np.argmax(silhouette_scores) + min_clusters
    return best_num_clusters

# Plot the best clustering result
def plot_best_cluster(X, best_num_clusters):
    custom_kmeans = CustomKMeans(n_clusters=best_num_clusters)
    custom_kmeans.fit(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=custom_kmeans.predict(X), cmap='viridis', marker='o', edgecolors='black')
    plt.title(f"Best Clustering Result (Number of Clusters: {best_num_clusters})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plot centroid movements for the best number of clusters
def plot_centroid_movements(X, centroids_history, best_num_clusters):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolors='black')

    colors = plt.cm.viridis(np.linspace(0, 1, len(centroids_history)))
    for i, centroids in enumerate(centroids_history):
        plt.scatter(centroids[:, 0], centroids[:, 1], c=[colors[i]], marker='x', s=200, label=f'Step {i + 1}')
        
    plt.title(f"Centroid Movements for {best_num_clusters} Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Find the best number of clusters based on silhouette score
best_num_clusters = find_best_n_clusters(X, min_clusters=2, max_clusters=8)
print(f"Best number of clusters: {best_num_clusters}")

# Plot the best clustering result
plot_best_cluster(X, best_num_clusters)

# Plot centroid movements for the best number of clusters
custom_kmeans = CustomKMeans(n_clusters=best_num_clusters)
custom_kmeans.fit(X)
plot_centroid_movements(X, custom_kmeans.centroids_history, best_num_clusters)  # Plot all steps
