import torch.nn as nn
import torch
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # TODO: This is fairly basic: can we do better?
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, input_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.randn(n_clusters, input_dim))  #init cluster centers randomly

    def forward(self, x):
        q = 1.0 / (1.0 + (torch.sum(torch.square(x.unsqueeze(1) - self.clusters), dim=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = F.normalize(q, p=1, dim=1)  #use softmax for normalization(similar to t-distribution)
        return q