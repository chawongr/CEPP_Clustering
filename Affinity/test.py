import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation
import numpy as np

# Generate complex blobs data
X, _ = make_blobs(n_samples=50, cluster_std=3.0, random_state=42)

class CustomAffinityPropagation:
    def __init__(self, damping=0.9, max_iter=200, convergence_iter=15):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.cluster_centers_ = None
        self.labels_history = []

    def fit(self, X):
        self.model = AffinityPropagation(
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter
        )
        self.labels_ = self.model.fit_predict(X)
        self.cluster_centers_ = X[self.model.cluster_centers_indices_]
        self.labels_history.append(self.labels_)

# Plot the clustering result for each step
def plot_affinity_propagation_step(X, exemplars, labels, step):
    plt.figure(figsize=(8, 6))
    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
        exemplar = X[exemplars[cluster_id]]
        plt.scatter(exemplar[0], exemplar[1], color='black', marker='x', s=100)
    plt.title(f"Affinity Propagation Clustering Step {step}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Fit the custom Affinity Propagation model and plot each step
custom_affinity_propagation = CustomAffinityPropagation()
# for step in range(custom_affinity_propagation.max_iter):
for step in range(4):

    custom_affinity_propagation.fit(X)
    exemplars = {i: np.where(custom_affinity_propagation.labels_ == i)[0][0] for i in range(len(np.unique(custom_affinity_propagation.labels_)))}
    plot_affinity_propagation_step(X, exemplars, custom_affinity_propagation.labels_, step)
