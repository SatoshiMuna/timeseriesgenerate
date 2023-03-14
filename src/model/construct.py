import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.network import device, dtype, vae_loss_function

class NetworkConstructor:
    """
    A class with reponsibility for training/testing the model.

    Parameters
    ----------
    model : Any - a time series generative network
    seed  : int - the randomizer seed
    """
    def __init__(self, model, seed=1):
        torch.manual_seed(seed)
        self.model = model
   
    def train(self, train_dataset, validate_dataset=None, batch_size=64, learning_rate=0.01, epoch=10):
        """
        Model training

        Parameters
        ----------
        train_dataset : StockSeriesDataSet     - the dataset for training
        validate_dataset : StockSeriesDataSet  - the dataset for validation
        batch_size : int                       - batch size
        learning_rate : float                  - learning rate
        epoch         : int                    - the number of training 
        """
        writer = SummaryWriter(log_dir='summary/'+type(self.model).__name__)
        logging.info('Start Training - size:%s, epoch:%s, batch:%s', len(train_dataset), epoch, batch_size)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, num_workers=2)
        statistics = train_dataset.col_stats
        self._do_train(train_loader, learning_rate, epoch, validate_dataset, writer, statistics)

        writer.close()

    def test(self, target_data, num_of_fcsts=200, latent_size=8):
        """
        Generate 1-step forecasts for target_data through decoder from latent vailables

        Parameters
        ----------
        target_data : Tensor - a time series whose 1step forecasts are generated  
        num_of_fcsts : int   - the number of forecasts generated
        latent_size : int     - the dimension of the latent valiable
        
        Returns
        -------
        out : list  - distribution of the 1-step forecasts 
        """
        logging.info('Start Testing - model:%s, num_of_fcsts:%s', type(self.model).__name__, num_of_fcsts)        
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_of_fcsts, latent_size).to(device, dtype)
            x = target_data[1].unsqueeze(0).to(device, dtype)
            out = []
            for i in range(z.size(0)):
                y = self.model.decode(x, z[i]).squeeze()
                out.append(y.to('cpu').numpy().copy())
        return out
    
    def _do_train(self, loader, lr, epoch, validate_dataset, writer, statistics):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.model.to(device, dtype)
        self.model.train()
        preloss = float('inf')
        for e in range(epoch):
            for idx, (x_ori, x_in) in enumerate(loader):
                # Calculate model output and loss
                x_in = x_in.to(device, dtype)
                x_ori = x_ori.to(device, dtype)
                x_out, mu, log_var = self.model(x_in)
                vae_loss, reconstruct_loss, kl_loss = vae_loss_function(x_out, x_ori[:,:,3:4], mu, log_var)
                # Backward and optimizer step
                optimizer.zero_grad()
                vae_loss.backward()
                optimizer.step()
            
            print(f"epoch:{e}, train_loss:{vae_loss.item()}, recon_loss:{reconstruct_loss.item()}, kl_loss:{kl_loss.item()}")
            logging.info('epoch:%s, train_loss:%s, recon_loss:%s, kl_loss:%s, model_save:%s', e, vae_loss.item(), reconstruct_loss.item(), kl_loss.item(), vae_loss.item()<preloss)
            if vae_loss.item() < preloss:
                torch.save(self.model.state_dict(), type(self.model).__name__+'_learned_model.pth')
                preloss = vae_loss.item()
            
            # Out-of-Sample Validation
            valid_loss, valid_recloss, valid_klloss = self._do_validate(validate_dataset)
            print(f"epoch:{e}, valid_loss:{valid_loss}, valid_recloss:{valid_recloss}, valid_klloss:{valid_klloss}")
            logging.info('epoch:%s, valid_loss:%s, recon_loss:%s, kl_loss:%s', e, valid_loss, valid_recloss, valid_klloss)
          
            # Output results
            self._write_results_for_tensorboard(writer, e+1, vae_loss.item(), reconstruct_loss.item(), kl_loss.item(), x_ori, x_out,
                                                valid_loss, valid_recloss, valid_klloss, validate_dataset[e], statistics)
                
    def _do_validate(self, dataset=None):
        validate_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for idx, (x_ori, x_in) in enumerate(validate_loader):
                # Calculate model output and loss
                x_in = x_in.to(device, dtype)
                x_ori = x_ori.to(device, dtype)
                x_out, mu, log_var = self.model(x_in)
                vae_loss, reconstruct_loss, kl_loss = vae_loss_function(x_out, x_ori[:,:,3:4], mu, log_var)
        return vae_loss.item(), reconstruct_loss.item(), kl_loss.item()

    def _write_results_for_tensorboard(self, writer, epoch, train_loss, train_rcloss, train_klloss, train_ori, train_out,
                                       test_loss, test_rcloss, test_klloss, test_sample, col_stats):
        writer.add_scalars('total_loss', {'train_loss':train_loss, 'validation_loss':test_loss}, epoch)
        writer.add_scalars('reconstruction_loss', {'train_recon':train_rcloss, 'validation_recon':test_rcloss}, epoch)
        writer.add_scalars('kl_loss', {'train_kl':train_klloss, 'validation_kl':test_klloss}, epoch)

        ori = train_ori[:,:,3].to('cpu').numpy().copy()
        gen = train_out.squeeze().to('cpu').detach().numpy().copy()
        stats = col_stats['Close']
        ori = (ori[0] * stats[1]) + stats[0]
        gen = (gen[0] * stats[1]) + stats[0]
        for i in range(len(ori)):
            writer.add_scalars('train-series/'+str(epoch), {'original':ori[i], 'generate':gen[i]}, i)
        self.model.eval()
        tout, _, _ = self.model(test_sample[1].unsqueeze(0).to(device, dtype))
        ori = test_sample[0][:,3].to('cpu').numpy().copy()
        gen = tout.squeeze().to('cpu').detach().numpy().copy()
        ori = (ori * stats[1]) + stats[0]
        gen = (gen * stats[1]) + stats[0]
        for i in range(len(ori)):
            writer.add_scalars('validation-series/'+str(epoch), {'original':ori[i], 'generate':gen[i]}, i)