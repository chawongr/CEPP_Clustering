import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Function to plot data and clusters
def plot_clusters(X, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

# Generate data
X, _ = make_blobs(n_samples=300, cluster_std=3.0)

# Range of cluster numbers to try
cluster_range = range(2, 8)

# Lists to store Calinski-Harabasz scores
ch_scores = []

# Lists to store cluster labels
all_labels = []

# Iterate over different cluster numbers
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)
    ch_score = calinski_harabasz_score(X, labels)
    ch_scores.append(ch_score)
    all_labels.append(labels)
    print(f"Clusters: {n_clusters}, Calinski-Harabasz Index: {ch_score}")

# Find the optimal number of clusters
best_n_clusters = max(zip(cluster_range, ch_scores), key=lambda x: x[1])[0]

# Perform clustering with optimal number of clusters
best_labels = all_labels[best_n_clusters - 2]

# Plot the results
plt.figure(figsize=(12, 6))

# Figure 1: Original data and clustered data
plt.subplot(1, 2, 1)
plot_clusters(X, None, title='Original Data')
plt.subplot(1, 2, 2)
plot_clusters(X, best_labels, title=f'KMeans Clustering (k={best_n_clusters})')

# Figure 2: Calinski-Harabasz Index
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, ch_scores, marker='o')
plt.title('Calinski-Harabasz Index for Different Cluster Numbers')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Index')

plt.tight_layout()
plt.show()
