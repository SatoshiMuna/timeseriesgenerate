import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from model.network import device, dtype, StockSeriesVAE, vae_loss_function
from data.dataset import StockSeriesDataSet

class NetworkTrainer:
    def __init__(self, stock_data, input_size=4, hidden_size=64, latent_size=16, sequence_len=32, target_len=1, insample_end_idx=None, seed=1):
        self.stock_data = stock_data
        self.sequence_length = sequence_len
        self.target_length = target_len
        self.insample_end_idx = insample_end_idx

        # Initialize network
        torch.manual_seed(seed)
        input_dim = self.sequence_length + self.target_length
        self.model = StockSeriesVAE(input_dim, hidden_size, latent_size)

    def do_train(self, batch_size=64, learning_rate=0.01, epoch=10):
        train_dataset = StockSeriesDataSet(True, self.stock_data, self.sequence_length, self.target_length, self.insample_end_idx)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
        # Start Model Training
        logging.info('Start Training - size:%s, epoch:%s, batch:%s', len(train_dataset), epoch, batch_size)
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.model.to(device, dtype)
        self.model.train()
        train_losses = []
        preloss = float('inf')
        for e in range(epoch):
            for idx, (x_ori, x_in) in enumerate(train_loader):
                # Calculate model output and loss
                x_in = x_in.to(device, dtype)
                x_ori = x_ori.to(device, dtype)
                x_out, mu, log_var = self.model(x_in)
                vae_loss, reconstruct_loss, kl_loss = vae_loss_function(x_out, x_ori, mu, log_var)
                # Backward and optimizer step
                optimizer.zero_grad()
                vae_loss.backward()
                optimizer.step()
            loss = vae_loss.item()
            train_losses.append(loss)
            logging.info('epoch:%s, loss:%s, recon_loss:%s, kl_loss:%s, model-save:%s', e, loss, reconstruct_loss.item(), kl_loss.item(), loss<preloss)
            if loss < preloss:
                torch.save(self.model.state_dict(), 'learned_model.pth')
                preloss = loss
            test_losses, test_reclosses, test_kllosses = self.do_test(train_dataset.col_stats)
            print(f"epoch:{e}, loss:{loss}, recon_loss:{reconstruct_loss.item()}, kl_loss:{kl_loss.item()}, test_loss:{test_losses.mean()}, test_recloss:{test_reclosses.mean()}")
            

    def do_test(self, col_stats):
        if self.model is None:
            self.model.load_state_dict(torch.load('learned_model.pth'))
        self.model.to(device, dtype)  
        self.model.eval()

        test_dataset = StockSeriesDataSet(False, self.stock_data, self.sequence_length, self.target_length, self.insample_end_idx, col_stats)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        loss = np.zeros(len(test_dataset))
        recloss = np.zeros(len(test_dataset))
        klloss = np.zeros(len(test_dataset))
        with torch.no_grad():
            for idx, (x_ori, x_in) in enumerate(test_loader):
                # Calculate model output and loss
                x_in = x_in.to(device, dtype)
                x_ori = x_ori.to(device, dtype)
                x_out, mu, log_var = self.model(x_in)
                vae_loss, reconstruct_loss, kl_loss = vae_loss_function(x_out, x_ori, mu, log_var)
                loss[idx] = vae_loss.item()
                recloss[idx] = reconstruct_loss.item()
                klloss[idx] = kl_loss.item()

        return loss, recloss, klloss