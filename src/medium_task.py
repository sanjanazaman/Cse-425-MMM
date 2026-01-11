import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


DATA_PATH = "../results/features_combined.npz"
SAVE_DIR = "../results/latent_visualization"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading features...")
data = np.load(DATA_PATH)
X = data["X"]
labels = data["y"]


mask = ~np.isnan(X).any(axis=1)
X = X[mask]
labels = labels[mask]

print(f"Samples: {X.shape[0]}, Feature dim: {X.shape[1]}")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)



kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
y_pred = kmeans.fit_predict(X_pca)

sil = silhouette_score(X_pca, y_pred)
ch = calinski_harabasz_score(X_pca, y_pred)

print("\nPCA + KMeans Metrics:")
print(f" - Silhouette Score: {sil:.4f}")
print(f" - Calinski-Harabasz Index: {ch:.4f}")


plt.figure(figsize=(8, 6))
for lbl in np.unique(y_pred):
    plt.scatter(X_pca[y_pred == lbl, 0],
                X_pca[y_pred == lbl, 1],
                s=5, label=f"Cluster {lbl}")
plt.legend()
plt.title("Medium Task: PCA + KMeans")
plt.savefig(f"{SAVE_DIR}/medium_pca_kmeans.png", dpi=300)
plt.close()

print("Plot saved to results/latent_visualization/")

