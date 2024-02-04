from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=10, centers=centers, cluster_std=0.5, random_state=42)

# Apply Affinity Propagation
af = AffinityPropagation(preference=-30).fit(X)

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# Plot connections between data points
plt.figure(figsize=(10, 6))
for i in range(X.shape[0]):
    plt.text(X[i, 0], X[i, 1], str(i), color=plt.cm.nipy_spectral(labels[i] / n_clusters_), fontdict={'weight': 'bold', 'size': 9})

for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        if af.affinity_matrix_[i, j]:
            plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], '-', color='gray', linewidth=0.2)

plt.xticks(np.arange(-2, 3, 1))
plt.yticks(np.arange(-2, 3, 1))
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Affinity Propagation: Step-by-Step Connection')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
