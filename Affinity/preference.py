import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation
from itertools import cycle

# Generate sample data without specifying cluster centers
# X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.8, random_state=42)
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=10, centers=centers, cluster_std=0.5, random_state=42)

# Perform Affinity Propagation clustering with different preferences
preferences = [-50, -30, -10, 0, 10]  # Example preference values

plt.figure(figsize=(15, 10))

for idx, preference in enumerate(preferences, 1):
    af = AffinityPropagation(preference=preference).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    plt.subplot(2, 3, idx)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('Preference = {}'.format(preference))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
