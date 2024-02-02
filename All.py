import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, cluster_std=5.0, random_state=42)

# Create a subplot for the original data
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolors='black')
plt.title("Original Data (No Clustering)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

kmeans = KMeans(n_clusters=3)
em = GaussianMixture(n_components=3)
dbscan = DBSCAN(eps=1.0, min_samples=5)
meanshift = MeanShift(bandwidth=2.0)
affinity_propagation = AffinityPropagation()
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
agglomerative = AgglomerativeClustering(n_clusters=3)  

algorithms = [kmeans, em, dbscan, meanshift, affinity_propagation, hdbscan_clusterer, agglomerative]
algorithm_names = ['KMeans', 'EM', 'DBSCAN', 'MeanShift', 'AffinityPropagation', 'HDBSCAN', 'Agglomerative']

plt.figure(figsize=(18, 12))

for i, algorithm in enumerate(algorithms, 1):
    # Fit the model
    if algorithm_names[i-1] != 'AffinityPropagation':
        labels = algorithm.fit_predict(X)
    else:
        labels = algorithm.fit(X).labels_

    # Plot the clusters
    plt.subplot(2, 4, i)  # Adjust the subplot grid to accommodate the new algorithm
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.title(algorithm_names[i-1])

plt.tight_layout()
plt.show()
