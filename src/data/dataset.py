import logging
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
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
    def __init__(self, is_train, stock_data, window_size, preprocess=True):
        super().__init__()
        self.inputs = []
        self.bc_lambda = None
        series = stock_data['Close']
        x = series.to_numpy()
        if preprocess:
            x, self.bc_lambda = self.preprocess(x)
        for i in range(x.shape[0] - window_size + 1):
            self.inputs.append(torch.tensor(x[i:i+window_size]))         
        
    def preprocess(self, x):
        # Box-Cox transform
        homogeneous_x, bc_lambda = boxcox(x)
        # Difference (remove trend)
        stationaly_x = np.diff(homogeneous_x)
        return stationaly_x, bc_lambda

    def get_boxcox_lambda(self):
        return self.bc_lambda

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]


    
