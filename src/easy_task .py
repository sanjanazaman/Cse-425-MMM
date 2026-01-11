# easy_task.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



FEATURES_FILE = "../results/features_combined.npz"
LATENT_VIS_DIR = "../results/latent_visualization/"


os.makedirs(LATENT_VIS_DIR, exist_ok=True)


print("Loading features...")
data = np.load(FEATURES_FILE)
X = data['X']
y = data['y']



mask = ~np.isnan(X).any(axis=1)
X, y = X[mask], y[mask]



scaler = StandardScaler()
X = scaler.fit_transform(X)


print(f"Total samples: {X.shape[0]}, Feature dim: {X.shape[1]}")



class VAE(nn.Module):
   def __init__(self, input_dim=44, latent_dim=8):
       super(VAE, self).__init__()
       self.fc1 = nn.Linear(input_dim, 32)
       self.fc_mu = nn.Linear(32, latent_dim)
       self.fc_logvar = nn.Linear(32, latent_dim)
       self.fc2 = nn.Linear(latent_dim, 32)
       self.fc3 = nn.Linear(32, input_dim)
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
       return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
   MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
   KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   return MSE + KLD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 8
vae = VAE(input_dim=X.shape[1], latent_dim=latent_dim).to(device)


X_tensor = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
epochs = 20


print("Training VAE...")
for epoch in range(1, epochs+1):
   vae.train()
   train_loss = 0
   for batch in loader:
       x_batch = batch[0].to(device)
       optimizer.zero_grad()
       recon, mu, logvar = vae(x_batch)
       loss = vae_loss(recon, x_batch, mu, logvar)
       loss.backward()
       optimizer.step()
       train_loss += loss.item()
   print(f"Epoch {epoch}/{epochs} | Loss: {train_loss/len(dataset):.4f}")


vae.eval()
with torch.no_grad():
   X_latent = vae.encode(torch.tensor(X, dtype=torch.float32).to(device))[0].cpu().numpy()



kmeans = KMeans(n_clusters=2, random_state=42)
y_pred = kmeans.fit_predict(X_latent)


sil_score = silhouette_score(X_latent, y_pred)
ch_score = calinski_harabasz_score(X_latent, y_pred)
print(f"\nKMeans Metrics:\n - Silhouette Score: {sil_score:.4f}\n - Calinski-Harabasz Index: {ch_score:.4f}")

pca = PCA(n_components=latent_dim)
X_pca = pca.fit_transform(X)
y_pca_pred = KMeans(n_clusters=2, random_state=42).fit_predict(X_pca)


sil_score_pca = silhouette_score(X_pca, y_pca_pred)
ch_score_pca = calinski_harabasz_score(X_pca, y_pca_pred)
print(f"\nPCA + KMeans Metrics:\n - Silhouette Score: {sil_score_pca:.4f}\n - Calinski-Harabasz Index: {ch_score_pca:.4f}")



tsne = TSNE(n_components=2, random_state=42)
tsne_latent = tsne.fit_transform(X_latent)


plt.figure(figsize=(8,6))
for lbl in np.unique(y):
   plt.scatter(tsne_latent[y==lbl,0], tsne_latent[y==lbl,1], label=str(lbl), s=5)
plt.legend()
plt.title("t-SNE visualization of VAE latent features")
plt.savefig(os.path.join(LATENT_VIS_DIR, "easy_task_tsne.png"))
plt.close()



plt.figure(figsize=(8,6))
for lbl in np.unique(y):
   plt.scatter(X_pca[y==lbl,0], X_pca[y==lbl,1], label=str(lbl), s=5)
plt.legend()
plt.title("PCA visualization of original features")
plt.savefig(os.path.join(LATENT_VIS_DIR, "easy_task_pca.png"))
plt.close()


