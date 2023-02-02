import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64 if torch.cuda.is_available() else torch.float32

def vae_loss_function(predict, target, mu, log_var):
    # negative log-likelihood
    reconstruct_loss = F.mse_loss(predict, target)     
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - torch.pow(mu, 2) - torch.exp(log_var), dim=1)
    kl_loss = torch.mean(kl_loss)
    vae_loss = reconstruct_loss + kl_loss
    return vae_loss, reconstruct_loss, kl_loss

class StockSeriesFcVAE(nn.Module):
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
        e = torch.randn_like(torch.exp(log_var)) # e～N(0, I)
        z = mu + torch.exp(log_var / 2) * e
        return mu, log_var, z

    def decode(self, z):
        # Decode latent variable 
        h = self.relu(self.fc_decode(z))
        # x = self.sigmoid(self.fc_output(h))
        x = self.fc_output(h)
        return x

    def forward(self, x):
        x = x.squeeze(2)
        mu, log_var, z = self.encode(x)
        x_decoded = self.decode(z)
        x_decoded = x_decoded.unsqueeze(2)
        return x_decoded, mu, log_var

class StockSeriesLstmVAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, latent_dim, sequence_length):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.D = 2 if bidirectional is True else 1
        self.sequence_length = sequence_length

        self.lstm_encode = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc_encode = nn.Linear(self.D*hidden_size, self.D*hidden_size)
        self.mu = nn.Linear(self.D*hidden_size, latent_dim)
        self.log_var = nn.Linear(self.D*hidden_size, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, sequence_length)#input_size*sequence_length)
        self.lstm_decode = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc_output = [nn.Linear(self.D*hidden_size, 1) for i in range(sequence_length)]

        self.relu = nn.ReLU()

    def encode(self, x):
        self.ohl = x[:,:,0:3]
        h_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size).to(device, dtype)
        c_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size).to(device, dtype)
        out, (h_n, c_n) = self.lstm_encode(x, (h_0, c_0))
        out = out[:,-1,:]                           # out:(N,L,D*Ho)
        h = self.relu(self.fc_encode(out))
        mu = self.mu(h)
        log_var = self.log_var(h) 

        # Obtain a value of latent valiable z
        eps = torch.randn_like(torch.exp(log_var))  # eps～N(0,1)
        z = mu + torch.exp(log_var / 2) * eps       # z～N(μ,σ)
        return mu, log_var, z

    def decode(self, z):
        z = self.relu(self.fc_decode(z))
        z = z.reshape(-1, self.sequence_length, 1)
        z = torch.cat((self.ohl, z), dim=2)
        h_0 = torch.zeros(self.D*self.num_layers, z.size(0), self.hidden_size).to(device, dtype)
        c_0 = torch.zeros(self.D*self.num_layers, z.size(0), self.hidden_size).to(device, dtype)
        out, (h_n, c_n) = self.lstm_decode(z, (h_0, c_0))  # out:(N,L,D*Ho)
        x = torch.zeros(z.size(0), self.sequence_length, 1)
        for i in range(self.sequence_length):
            x[:,i,:] = self.fc_output[i](out[:,i,:])       # x:(N,1)
        # x = self.fc_output(out)                            # x:(N,L)
        return x

    def forward(self, x):
        mu, log_var, z = self.encode(x)
        x_decoded = self.decode(z)
        return x_decoded, mu, log_var

class StockSeriesLstmVAE2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, latent_dim, sequence_length):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.D = 2 if bidirectional is True else 1
        self.lstm_encode = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc_encode = nn.Linear(self.D*hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_log_var = nn.Linear(hidden_size, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, hidden_size)
        self.lstm_decode = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc_decode2 = nn.Linear(hidden_size, input_size)
        self.fc_output = nn.Linear(self.D*hidden_size, 1)
        self.relu = nn.GELU()

    def encode(self, x):
        h_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size).to(device, dtype)
        c_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size).to(device, dtype)
        out, (h_n, c_n) = self.lstm_encode(x, (h_0, c_0))  # out:(N,L,D*Ho)
        out = out[:,-1,:]                                  # out:(N,D*Ho)
        out = self.relu(self.fc_encode(out))
        mu = self.fc_mu(out)
        log_var = self.fc_log_var(out)

        eps = torch.randn_like(torch.exp(log_var))
        z = mu + torch.exp(log_var/2) * eps
        return mu, log_var, z        

    def decode(self, x, z):
        z = self.relu(self.fc_decode(z))
        z = self.fc_decode2(z)
        x[:,-1,:] = z
        h_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size).to(device, dtype)
        c_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size).to(device, dtype)
        out, (h_n, c_n) = self.lstm_decode(x, (h_0, c_0))   # out:(N,L,D*Ho)
        val = self.fc_output(out[:,-1,:])
        x[:,-1,3] = val.squeeze(1)
        return x[:,:,3:4]

    def forward(self, x):
        mu, log_var, z = self.encode(x)
        x_decoded = self.decode(x, z)
        return x_decoded, mu, log_var