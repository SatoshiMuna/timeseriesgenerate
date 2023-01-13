import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64 if torch.cuda.is_available() else torch.float32

def vae_loss_function(predict, target, mu, log_var):
    reconstruct_loss = F.binary_cross_entropy(predict, target, reduction='sum')     # negative log-likelihood
    kl_loss = -0.5 * torch.sum(1 + log_var - torch.pow(mu, 2) - torch.exp(log_var)) # KL divergence
    vae_loss = reconstruct_loss + kl_loss
    return vae_loss, reconstruct_loss, kl_loss

class StockSeriesVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc_encode = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        # Encode input
        h = self.relu(self.fc_encode(x))
        mu = self.fc_mu(h)       # μ(φ)
        log_var = self.fc_var(h) # log(σ^2)
        
        # Obtain the latent valiable z = μ + σ・ε
        e = torch.rand_like(torch.exp(log_var)) # e～N(0, I)
        z = mu + torch.exp(log_var / 2) * e

        return mu, log_var, z

    def decode(self, z):
        # Decode latent variable 
        h = self.relu(self.fc_decode(z))
        x = self.sigmoid(self.fc_output(h))
        return x

    def forward(self, x):
        mu, log_var, z = self.encode(x)
        x_decoded = self.decode(z)
        return x_decoded, mu, log_var