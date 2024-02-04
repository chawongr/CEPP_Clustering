import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation
import numpy as np

# Generate complex blobs data
X, _ = make_blobs(n_samples=500, cluster_std=5.0, random_state=42)

class CustomAffinityPropagation:
    def __init__(self, damping=0.9, max_iter=200, convergence_iter=15):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.cluster_centers_ = None
        self.labels_ = None
        self.centers_history = []
        self.members_history = []

    def fit(self, X):
        self.model = AffinityPropagation(
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter
        )
        self.labels_ = self.model.fit_predict(X)
        self.cluster_centers_ = X[self.model.cluster_centers_indices_]
        self.centers_history.append(self.cluster_centers_)
        self.members_history.append(self.labels_)

# Fit the custom Affinity Propagation model
custom_affinity_propagation = CustomAffinityPropagation()
for step in range(custom_affinity_propagation.max_iter):
    custom_affinity_propagation.fit(X)
    # Check for convergence
    if step > 0 and np.all(custom_affinity_propagation.labels_ == previous_labels):
        print(f"Converged after {step} iterations.")
        break
    previous_labels = custom_affinity_propagation.labels_.copy()

# Print center and its members
for i, (centers, members) in enumerate(zip(custom_affinity_propagation.centers_history, custom_affinity_propagation.members_history)):
    print(f"Step {i}:")
    for center_index, center in enumerate(centers):
        members_indices = np.where(members == center_index)[0]
        print(f"Center {center_index}: {center} has members {members_indices}")

# Plot the final clustering result
plt.figure(figsize=(8, 6))
colors = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(custom_affinity_propagation.labels_))))
for label, color in zip(np.unique(custom_affinity_propagation.labels_), colors):
    cluster_mask = custom_affinity_propagation.labels_ == label
    plt.scatter(X[cluster_mask, 0], X[cluster_mask, 1], color=color, marker='o', edgecolors='black', label=f'Cluster {label}')
plt.scatter(custom_affinity_propagation.cluster_centers_[:, 0], custom_affinity_propagation.cluster_centers_[:, 1],
            color='black', marker='x', s=100, label='Cluster Centers')
plt.title("Final Affinity Propagation Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
