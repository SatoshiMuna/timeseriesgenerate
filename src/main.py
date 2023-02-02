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
    fcst_open2close = True  # if True, input_size is set to 5
    trainer = NetworkTrainer(stock_data=stock_data, input_size=5, hidden_size=128, latent_size=16, sequence_len=32,
                             bidirectional=False, insample_end_idx=insample_end_idx, open2close=fcst_open2close)
    if exec_training == 'y':
        trainer.do_train(epoch=50)
    else:
        loss, recloss, klloss, sample, stats = trainer.do_test(isTestOnly=True, index=3)
        model = trainer.get_model()          
        model.eval()
        with torch.no_grad():
            z = torch.randn(200, 16).to(device, dtype)
            x = sample[1].unsqueeze(0).to(device, dtype)
            y = []
            for i in range(z.size(0)):
                out = model.decode(x, z[i]).squeeze()
                y.append(out.to('cpu').numpy().copy())
            ori = sample[0][:,3][-1].to('cpu').numpy().copy()
            pre = sample[0][:,3][-2].to('cpu').numpy().copy()
        _visualize(y, stats['Close'], ori, pre)

def _visualize(values, stats, original, previous):
        x = [s[-1]*stats[1]+stats[0] for s in values]
        original = original * stats[1] + stats[0]
        previous = previous * stats[1] + stats[0]
        fig, axes = plt.subplots(1,2)
        v0 = axes[0].hist(x, bins=20, alpha=0.5)
        axes[0].vlines(original, ymin=0, ymax=v0[0].max(), colors='b', label='next price')
        axes[0].vlines(previous, ymin=0, ymax=v0[0].max(), colors='r', label='current price')
        axes[0].set_title('forecast distribution')
        axes[0].set_xlabel('close price')
        axes[0].set_ylabel('frequence')
        axes[0].legend()
        axes[0].grid(True)
        v1 = axes[1].hist(x, bins=20, alpha=0.5, cumulative=True, density=True)
        axes[1].vlines(previous, ymin=0, ymax=v1[0].max(), colors='r', label='current price')
        axes[1].set_title('forecast cumulative distribution')
        axes[1].set_xlabel('close price')
        axes[1].set_ylabel('probability')
        axes[1].legend()
        axes[1].grid(True)
        plt.show()

if __name__ == '__main__':
    main('6501.T','2012-01-01','2022-12-01','2022-01-31','n')