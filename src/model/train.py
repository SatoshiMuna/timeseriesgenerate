import torch
from torch.utils.data import DataLoader
from model.network import device, dtype, StockSeriesVAE, vae_loss_function
from data.dataset import StockSeriesDataSet

class NetworkTrainer:
    def __init__(self, stock_data, input_size=4, hidden_size=32, latent_size=16, sequence_len=32, target_len=1, preprocess=True, seed=1):
        self.stock_data = stock_data
        self.sequence_length = sequence_len
        self.target_length = target_len
        self.preprocess = preprocess

        # Initialize network
        torch.manual_seed(seed)
        input_dim = self.sequence_length + self.target_length
        self.model = StockSeriesVAE(input_dim, hidden_size, latent_size)

    def do_train(self, batch_size=64, learning_rate=0.01, epoch=10):
        train_dataset = StockSeriesDataSet(True, self.stock_data, self.sequence_length, self.target_length, self.preprocess)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
        # Start Model Training
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.model.to(device, dtype)
        self.model.train()
        train_losses = []
        for e in range(epoch):
            vae_loss = None
            for idx, (x, t, t_ori) in enumerate(train_loader):
                x_in = torch.cat((x, t), dim=1)
                x_ori = torch.cat((x, t_ori), dim=1)
                x_in = x_in.to(device, dtype)
                x_ori = x_ori.to(device, dtype)
                x_out, mu, log_var = self.model(x_in)
                vae_loss, reconstruct_loss, kl_loss = vae_loss_function(x_out, x_ori, mu, log_var)
                # Backward and optimize step
                optimizer.zero_grad()
                vae_loss.backward()
                optimizer.step()
            train_losses.append(vae_loss.item())
            print(f"epoch:{e}, loss:{vae_loss.item()}, recon_loss:{reconstruct_loss.item()}, kl_loss:{kl_loss.item()}")
