import torch
from torch.utils.data import DataLoader
from model.network import device, dtype, StockSeriesVAE, vae_loss_function
from data.dataset import StockSeriesDataSet


class NetworkTrainer:
    def __init__(self, stock_data, input_dim=32, hidden_dim=32, latent_dim=16, window_size=32, preprocess=True, seed=1):
        self.stock_data = stock_data
        self.window_size = window_size
        self.preprocess = preprocess

        # Initialize network
        torch.manual_seed(seed)
        self.model = StockSeriesVAE(input_dim, hidden_dim, latent_dim)

    def do_train(self, batch_size=64, learning_rate=0.01, epoch=10):
        train_dataset = StockSeriesDataSet(True, self.stock_data, self.window_size, self.preprocess)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
        # Start Model Training
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.model.to(device, dtype)
        self.model.train()
        train_losses = []
        for e in range(epoch):
            vae_loss = None
            for idx, x in enumerate(train_loader):
                x = x.to(device, dtype)
                x_pred, mu, log_var = self.model(x)
                vae_loss, reconstruct_loss, kl_loss = vae_loss_function(x_pred, x, mu, log_var)

                optimizer.zero_grad()
                vae_loss.backward()
                optimizer.step()
            train_losses.append(vae_loss.item())
            print(f"epoch:{e}, loss:{vae_loss.item()}, recon_loss:{reconstruct_loss.item()}, kl_loss:{kl_loss.item()}")