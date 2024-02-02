# import numpy as np

# # สร้าง array ที่ไม่ได้เรียงต่อกัน
# non_contig_array = np.array([[1, 2, 3], [4, 5, 6]])
# print(non_contig_array)
# # แปลงเป็น C-contiguous array
# contig_array = np.ascontiguousarray(non_contig_array)
# print(contig_array)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import time

# สร้างข้อมูล
X, y = make_blobs(n_samples=500, centers=3, cluster_std=5.0, random_state=42)

# ไม่ใช้ np.ascontiguousarray
start_time = time.time()
kmeans_no_contig = KMeans(n_clusters=3, random_state=42)
labels_no_contig = kmeans_no_contig.fit_predict(X)
end_time = time.time()
print(f"Time without np.ascontiguousarray: {end_time - start_time:.6f} seconds")

# ใช้ np.ascontiguousarray
X_contig = np.ascontiguousarray(X)
start_time = time.time()
kmeans_contig = KMeans(n_clusters=3, random_state=42)
labels_contig = kmeans_contig.fit_predict(X_contig)
end_time = time.time()
print(f"Time with np.ascontiguousarray: {end_time - start_time:.6f} seconds")
