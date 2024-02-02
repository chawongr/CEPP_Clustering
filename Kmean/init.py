# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# from sklearn.metrics import silhouette_score

# # Generate data
# X, y = make_blobs(n_samples=500,centers=3, cluster_std=5.0, random_state=42)

# # Create a subplot for the original data
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolors='black')
# plt.title("Original Data (No Clustering)")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")

# # Initialize KMeans with 'random' initialization
# kmeans_random = KMeans(n_clusters=3, init='random', random_state=42)
# labels_random = kmeans_random.fit_predict(X)

# # Plot the clusters using 'random' initialization
# plt.subplot(1, 2, 2)
# plt.scatter(X[:, 0], X[:, 1], c=labels_random, cmap='viridis', edgecolor='k', s=50)
# plt.title("KMeans with init='random'")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")

# # Display silhouette score for 'random' initialization
# silhouette_random = silhouette_score(X, labels_random)
# print(f"Silhouette Score with 'random' initialization: {silhouette_random}")

# plt.tight_layout()
# plt.show()

# # Initialize KMeans with 'k-means++' initialization
# kmeans_kmeans_plus_plus = KMeans(n_clusters=3, init='k-means++', random_state=42)
# labels_kmeans_plus_plus = kmeans_kmeans_plus_plus.fit_predict(X)

# # Plot the clusters using 'k-means++' initialization
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans_plus_plus, cmap='viridis', edgecolor='k', s=50)
# plt.title("KMeans with init='k-means++'")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")

# # Display silhouette score for 'k-means++' initialization
# silhouette_kmeans_plus_plus = silhouette_score(X, labels_kmeans_plus_plus)
# print(f"Silhouette Score with 'k-means++' initialization: {silhouette_kmeans_plus_plus}")

# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Generate data with clusters of different variances
X, y = make_blobs(n_samples=500, centers=3, cluster_std=[2.0, 4.0, 6.0], random_state=42)

# Create a subplot for the original data
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolors='black')
plt.title("Original Data (No Clustering)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Initialize KMeans with 'random' initialization
kmeans_random = KMeans(n_clusters=5, init='random', random_state=42)
labels_random = kmeans_random.fit_predict(X)

# Plot the clusters using 'random' initialization
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_random, cmap='viridis', edgecolor='k', s=50)
plt.title("KMeans with init='random'")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Display silhouette score for 'random' initialization
silhouette_random = silhouette_score(X, labels_random)
print(f"Silhouette Score with 'random' initialization: {silhouette_random}")

plt.tight_layout()
plt.show()

# Initialize KMeans with 'k-means++' initialization
kmeans_kmeans_plus_plus = KMeans(n_clusters=5, init='k-means++', random_state=42)
labels_kmeans_plus_plus = kmeans_kmeans_plus_plus.fit_predict(X)

# Plot the clusters using 'k-means++' initialization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans_plus_plus, cmap='viridis', edgecolor='k', s=50)
plt.title("KMeans with init='k-means++'")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Display silhouette score for 'k-means++' initialization
silhouette_kmeans_plus_plus = silhouette_score(X, labels_kmeans_plus_plus)
print(f"Silhouette Score with 'k-means++' initialization: {silhouette_kmeans_plus_plus}")

plt.tight_layout()
plt.show()
