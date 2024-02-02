import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# สร้างข้อมูลจุด (blobs) ที่ซับซ้อนมากขึ้น
X, _ = make_blobs(n_samples=500, cluster_std=5.0, random_state=42)

# ทดลองใช้ KMeans แบ่งกลุ่มข้อมูลด้วยจำนวนกลุ่มที่เปลี่ยนไป
min_clusters, max_clusters = 2, 8
silhouette_scores = []

# พล็อตข้อมูลตอนแรก
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolors='black')
plt.title("Original Data (No Clustering)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# คำนวณ Silhouette Score สำหรับแต่ละจำนวนกลุ่ม
for n_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")

# แสดงกราฟ Silhouette Score สำหรับแต่ละจำนวนกลุ่ม
plt.subplot(1, 2, 2)
plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

# เลือกจำนวนกลุ่มที่ให้ Silhouette Score รวมสูงที่สุด
best_num_clusters = np.argmax(silhouette_scores) + min_clusters
best_silhouette_score = max(silhouette_scores)
print(f"Best number of clusters: {best_num_clusters} with Silhouette Score: {best_silhouette_score}")

# ทดลองใช้ KMeans จัดกลุ่มข้อมูลด้วยจำนวนกลุ่มที่เลือก
best_kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
best_labels = best_kmeans.fit_predict(X)

# พล็อตข้อมูลหลังจากจัดกลุ่มด้วยจำนวนกลุ่มที่เลือก
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolors='black')
plt.title("Original Data (No Clustering)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=best_labels, cmap='viridis', marker='o', edgecolors='black')
plt.title(f"Data Clustered by KMeans (Best Number of Clusters: {best_num_clusters})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
