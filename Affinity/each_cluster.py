from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)

# Apply Affinity Propagation
af = AffinityPropagation(preference=-50).fit(X)

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

# Plot result with movement
for k in range(n_clusters_):
    plt.figure(k+1)
    plt.clf()

    exemplars = X[af.labels_ == k]
    cluster_center = X[cluster_centers_indices[k]]

    plt.scatter(X[:, 0], X[:, 1], color='black', s=7)
    plt.plot(exemplars[:, 0], exemplars[:, 1], 'o', markerfacecolor='blue', markersize=3)
    plt.plot(exemplars[:, 0], exemplars[:, 1], '-', color='gray', linewidth=0.2)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor='orange', markersize=12)

    plt.title('Cluster %d' % (k+1))

plt.show()
