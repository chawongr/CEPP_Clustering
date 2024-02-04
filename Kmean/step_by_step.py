import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.metrics import silhouette_score

# Generate complex blobs data
X, _ = make_blobs(n_samples=500, cluster_std=3.0, random_state=42)

class CustomKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids_history = []
        self.labels_history = []

    def fit(self, X):
        # Randomly initialize centroids
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            # Assign each data point to the nearest centroid
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)
            self.labels_history.append(labels)

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

# Plot the clustering result for each step
def plot_clustering_step(X, centroids, labels, step):
    plt.figure(figsize=(8, 6))
    for cluster_id in range(len(centroids)):
        cluster_points = X[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
        plt.scatter(centroids[cluster_id, 0], centroids[cluster_id, 1], color='black', marker='x', s=100)
    plt.title(f"Clustering Step {step}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Fit the custom KMeans model and plot each step
custom_kmeans = CustomKMeans(n_clusters=3)
custom_kmeans.fit(X)
for step, (centroids, labels) in enumerate(zip(custom_kmeans.centroids_history, custom_kmeans.labels_history)):
    plot_clustering_step(X, centroids, labels, step)
