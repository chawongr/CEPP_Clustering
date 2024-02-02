from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate data
X, y = make_blobs(n_samples=500, centers=3, cluster_std=5.0, random_state=42)

# Set verbose=0
kmeans_verbose_0 = KMeans(n_clusters=3, verbose=0)
labels_verbose_0 = kmeans_verbose_0.fit_predict(X)

# Set verbose=1
kmeans_verbose_1 = KMeans(n_clusters=3, verbose=1)
labels_verbose_1 = kmeans_verbose_1.fit_predict(X)

# # Set verbose=2
# kmeans_verbose_2 = KMeans(n_clusters=3, verbose=2)
# labels_verbose_2 = kmeans_verbose_2.fit_predict(X)
