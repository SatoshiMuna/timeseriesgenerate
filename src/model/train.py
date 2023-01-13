import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from network import device, dtype, StockSeriesVAE, vae_loss_function
from data.dataset import StockSeriesDataSet


class NetworkTrainer:
    def __init__(self, stock_data, input_dim=32, hidden_dim=32, latent_dim=16, window_size=32):
        self.stock_data = stock_data
        self.window_size = window_size
        self.model = StockSeriesVAE(input_dim, hidden_dim, latent_dim)

    def do_train(self, batch_size=32, learning_rate=0.01, epoch=30):
        train_dataset = StockSeriesDataSet(True, self.stock_data, self.window_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

        optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)

        self.model.to(device, dtype)
        self.model.train()
        train_losses = []
        for e in range(epoch):
            loss = None
            for idx, (x, y) in enumerate(train_loader):
                x.to(device, dtype)
                y.to(device, dtype)
                y_pred, mu, log_var = self.model(x)
                vae_loss, reconstruct_loss, kl_loss = vae_loss_function(y_pred, y, mu, log_var)

                optimizer.zero_grad()
                vae_loss.backward()
                optimizer.step()
            train_losses.append(loss.item())
            print(f"epoch:{e}, loss:{loss.item()}")