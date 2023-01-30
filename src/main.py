import logging
import torch
import matplotlib.pyplot as plt

from model.train import NetworkTrainer
from data.dataset import get_stock_data
from model.network import device, dtype

def main(stock_code, start_date, end_date, insample_end_date, exec_training):
    logging.basicConfig(filename='timeseriesgenerate.log', level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    stock_data = get_stock_data(stock_code, start_date, end_date)
    insample_end_idx = stock_data.index.get_loc(insample_end_date)
    trainer = NetworkTrainer(stock_data=stock_data, input_size=4, hidden_size=128, latent_size=8, insample_end_idx=insample_end_idx)
    if exec_training == 'y':
        trainer.do_train(epoch=100)
    else:
        loss, recloss, klloss, sample, stats = trainer.do_test(isTestOnly=True, index=30)
        model = trainer.get_model()          
        model.eval()
        with torch.no_grad():
            z = torch.randn(100, 8).to(device, dtype)
            x = sample[1].unsqueeze(0).to(device, dtype)
            y = []
            for i in range(z.size(0)):
                out = model.decode(x, z[i]).squeeze()
                y.append(out.to('cpu').numpy().copy())
            ori = sample[0][:,3][-1].to('cpu').numpy().copy()
            pre = sample[0][:,3][-2].to('cpu').numpy().copy()
        st = stats['Close']
        c = [s[-1]*st[1]+st[0] for s in y]
        ori = ori*st[1]+st[0]
        pre = pre*st[1]+st[0]
        plt.grid(True)
        v = plt.hist(c, bins=16, alpha=0.5)
        plt.vlines(ori, v[0].min(), v[0].max())
        plt.vlines(pre, v[0].min(), v[0].max(), 'r')
        plt.show()

if __name__ == '__main__':
    main('6501.T','2012-01-01','2022-12-01','2022-01-31','n')