import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


FEATURES_FILE = "../results/features_combined.npz"
LATENT_VIS_DIR = "../results/latent_visualization/"
CLUSTER_CSV = "../results/clustering_metrics.csv"

BATCH_SIZE = 128
LATENT_DIM = 16
HIDDEN_DIM = 64
EPOCHS = 20
LEARNING_RATE = 1e-3


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
      
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
 
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def save_latent_visualizations(latent_features, labels, task_name="hard"):
    os.makedirs(LATENT_VIS_DIR, exist_ok=True)
    
  
    tsne = TSNE(n_components=2, random_state=42)
    tsne_latent = tsne.fit_transform(latent_features)
    plt.figure(figsize=(8,6))
    for lbl in np.unique(labels):
        plt.scatter(tsne_latent[labels==lbl,0], tsne_latent[labels==lbl,1], label=str(lbl), s=5)
    plt.legend()
    plt.title(f"t-SNE visualization ({task_name} task)")
    plt.savefig(os.path.join(LATENT_VIS_DIR, f"{task_name}_tsne.png"))
    plt.close()


def main():
 
    print("Loading features...")
    data = np.load(FEATURES_FILE)
    X = data['X']
    y = data['y']


    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    print(f"Total samples: {X.shape[0]}, Feature dim: {X.shape[1]}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    
    vae = VAE(input_dim=X.shape[1], hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)


    print("Training VAE...")
    vae.train()
    for epoch in range(1, EPOCHS+1):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x_batch = batch[0]
            recon, mu, logvar = vae(x_batch)
            loss = vae_loss(recon, x_batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss / X.shape[0]:.4f}")

    
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(torch.tensor(X, dtype=torch.float32))
        latent_features = mu.numpy()

    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)
    y_pred = kmeans.fit_predict(latent_features)
    sil_score = silhouette_score(latent_features, y_pred)
    ch_score = calinski_harabasz_score(latent_features, y_pred)
    print("\nKMeans Metrics:")
    print(f" - Silhouette Score: {sil_score:.4f}")
    print(f" - Calinski-Harabasz Index: {ch_score:.4f}")

    
    X_pca = PCA(n_components=LATENT_DIM).fit_transform(X)
    y_pred_pca = KMeans(n_clusters=len(np.unique(y)), random_state=42).fit_predict(X_pca)
    sil_pca = silhouette_score(X_pca, y_pred_pca)
    ch_pca = calinski_harabasz_score(X_pca, y_pred_pca)
    print("\nPCA + KMeans Metrics:")
    print(f" - Silhouette Score: {sil_pca:.4f}")
    print(f" - Calinski-Harabasz Index: {ch_pca:.4f}")

    save_latent_visualizations(latent_features, y, task_name="hard")
    print(f"Plot saved to {LATENT_VIS_DIR}")

    
    metrics = pd.DataFrame({
        "Method": ["KMeans(VAE)", "KMeans(PCA)"],
        "Silhouette": [sil_score, sil_pca],
        "Calinski-Harabasz": [ch_score, ch_pca]
    })
    metrics.to_csv(CLUSTER_CSV, index=False)
    print(f"Clustering metrics saved to {CLUSTER_CSV}")

if __name__ == "__main__":
    main()
