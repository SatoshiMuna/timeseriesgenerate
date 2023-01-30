import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.network import device, dtype, StockSeriesFcVAE, StockSeriesLstmVAE, StockSeriesLstmVAE2, vae_loss_function
from data.dataset import StockSeriesDataSet

class NetworkTrainer:
    def __init__(self, stock_data, input_size=4, hidden_size=128, latent_size=8, sequence_len=32, target_len=1,
                 num_layers=1, bidirectional=False, insample_end_idx=None, seed=1):
        self.stock_data = stock_data
        self.sequence_length = sequence_len
        self.target_length = target_len
        self.insample_end_idx = insample_end_idx

        # Initialize network
        torch.manual_seed(seed)
        input_dim = self.sequence_length + self.target_length
        #self.model = StockSeriesFcVAE(input_dim, hidden_size, latent_size)
        #self.model = StockSeriesLstmVAE(input_size, hidden_size, num_layers, bidirectional, latent_size, sequence_len+target_len)
        self.model = StockSeriesLstmVAE2(input_size, hidden_size, num_layers, bidirectional, latent_size, sequence_len+target_len)           
   
    def do_train(self, batch_size=64, learning_rate=0.01, epoch=10):
        writer = SummaryWriter(log_dir='summary/'+type(self.model).__name__)
        # Training Data
        train_dataset = StockSeriesDataSet(True, self.stock_data, self.sequence_length, self.target_length, self.insample_end_idx)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
        # Start Model Training
        logging.info('Start Training - size:%s, epoch:%s, batch:%s', len(train_dataset), epoch, batch_size)
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.model.to(device, dtype)
        self.model.train()
        preloss = float('inf')
        for e in range(epoch):
            for idx, (x_ori, x_in) in enumerate(train_loader):
                # Calculate model output and loss
                x_in = x_in.to(device, dtype)
                x_ori = x_ori.to(device, dtype)
                x_out, mu, log_var = self.model(x_in)
                vae_loss, reconstruct_loss, kl_loss = vae_loss_function(x_out, x_ori[:,:,3:], mu, log_var)
                # Backward and optimizer step
                optimizer.zero_grad()
                vae_loss.backward()
                optimizer.step()
            
            print(f"epoch:{e}, train_loss:{vae_loss.item()}, recon_loss:{reconstruct_loss.item()}, kl_loss:{kl_loss.item()}")
            logging.info('epoch:%s, train_loss:%s, recon_loss:%s, kl_loss:%s, model_save:%s', e, vae_loss.item(), reconstruct_loss.item(), kl_loss.item(), vae_loss.item()<preloss)
            if vae_loss.item() < preloss:
                torch.save(self.model.state_dict(), type(self.model).__name__+'_learned_model.pth')
                preloss = vae_loss.item()
            
            # Out-of-Sample Testing
            test_losses, test_reclosses, test_kllosses, t_in = self.do_test(train_dataset.col_stats, e)
            print(f"epoch:{e}, test_loss:{test_losses.mean()}, test_recloss:{test_reclosses.mean()}, test_klloss:{test_kllosses.mean()}")
            
            # Output results
            self._write_results_for_tensorboard(writer, e+1, vae_loss.item(), reconstruct_loss.item(), kl_loss.item(), x_ori, x_out,
                                                test_losses.mean(), test_reclosses.mean(), test_kllosses.mean(), t_in, train_dataset.col_stats)
                
        writer.close()
  
    def do_test(self, col_stats=None, index=0):
        if self.model is None:
            train_dataset = StockSeriesDataSet(True, self.stock_data, self.sequence_length, self.target_length, self.insample_end_idx)
            col_stats = train_dataset.col_stats
            self.model.load_state_dict(torch.load(type(self.model).__name__+'_learned_model.pth'))            
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
                vae_loss, reconstruct_loss, kl_loss = vae_loss_function(x_out, x_ori[:,:,3:], mu, log_var)
                loss[idx] = vae_loss.item()
                recloss[idx] = reconstruct_loss.item()
                klloss[idx] = kl_loss.item()
        
        return loss, recloss, klloss, test_dataset[index]


    def _write_results_for_tensorboard(self, writer, epoch, train_loss, train_rcloss, train_klloss, train_ori, train_out,
                                       test_loss, test_rcloss, test_klloss, test_sample, col_stats):
        writer.add_scalars('total_loss', {'train_loss':train_loss, 'test_loss':test_loss}, epoch)
        writer.add_scalars('reconstruction_loss', {'train_recon':train_rcloss, 'test_recon':test_rcloss}, epoch)
        writer.add_scalars('kl_loss', {'train_kl':train_klloss, 'test_kl':test_klloss}, epoch)
        ori = train_ori[:,:,3].to('cpu').numpy().copy()
        gen = train_out.to('cpu').detach().numpy().copy()
        stats = col_stats['Close']
        ori = (ori[0] * stats[1]) + stats[0]
        gen = (gen[0] * stats[1]) + stats[0]
        for i in range(len(ori)):
            writer.add_scalars('train-series/'+str(epoch), {'original':ori[i], 'generate':gen[i]}, i)
        
        tout, _, _ = self.model(test_sample[1].unsqueeze(0).to(device, dtype))
        ori = test_sample[0].unsqueeze(0)[:,:,3].to('cpu').numpy().copy()
        gen = tout.to('cpu').detach().numpy().copy()
        ori = (ori[0] * stats[1]) + stats[0]
        gen = (gen[0] * stats[1]) + stats[0]
        for i in range(len(ori)):
            writer.add_scalars('test-series/'+str(epoch), {'original':ori[i], 'generate':gen[i]}, i)