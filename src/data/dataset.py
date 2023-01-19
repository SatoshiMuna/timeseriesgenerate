import logging
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data.datautil import Transform
from scipy.stats import boxcox
from scipy.special import inv_boxcox

def get_stock_data(stock_code, start_date, end_date, use_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
    code_info = yf.Ticker(stock_code) 
    start = start_date
    end = end_date
    # Return pandas.DataFrame(date, stock_values)
    stock_data = code_info.history(start=start, end=end)
    logging.info('Get stock data - code:%s, data start:%s, data end:%s', stock_code, start_date, end_date)
    return stock_data[use_cols]
    
class StockSeriesDataSet(Dataset):
    def __init__(self, is_train, stock_data, window_size, target_size, insample_end_idx, col_stats=None, mask=0):
        super().__init__()
        self.inputs = []
        self.targets = []
        self.col_stats = {} if col_stats is None else col_stats
        series = stock_data[['Close']]
        
        if is_train:
            x = series[:insample_end_idx+1]
            for col_name in x:
                self.col_stats[col_name] = (x.loc[:,col_name].mean(), x.loc[:,col_name].std())
            x = x.apply(lambda z: (z-z.mean())/z.std(), axis=0)                
        else:
            x = series[insample_end_idx-window_size:].copy()
            for col_name in x:
                stats = self.col_stats[col_name]
                x.loc[:, col_name] = x.loc[:,col_name].apply(lambda z: (z-stats[0])/stats[1])
                
        x = x.to_numpy()
        for i in range(x.shape[0] - window_size + 1 - target_size):
            self.inputs.append(np.squeeze(x[i:i+window_size+target_size]))
            t = np.copy(x[i:i+window_size+target_size]) 
            t[-target_size:,:] = mask           
            self.targets.append(np.squeeze(t))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]), torch.tensor(self.targets[index])


    
