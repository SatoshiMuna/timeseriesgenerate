import logging
import yfinance as yf
import numpy as np
import copy
import torch

from torch.utils.data import Dataset

def get_stock_data(stock_code, start_date, end_date, use_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
    code_info = yf.Ticker(stock_code) 
    start = start_date
    end = end_date
    # Return pandas.DataFrame(date, stock_values)
    stock_data = code_info.history(start=start, end=end)
    logging.info('Get stock data - code:%s, data start:%s, data end:%s', stock_code, start_date, end_date)
    return stock_data[use_cols]
    
class StockSeriesDataSet(Dataset):
    """
    DataSet class
    """
    def __init__(self, stock_data, window_size, target_size, insample_end_idx, use_cols=['Open', 'High', 'Low', 'Close'],
                 open2close=False, is_train=True, mask=-100):
        """
        Constructor

        Parameters
        ----------
        stock_data : Pandas.DataFrame - the matrix data for various stock values time series
        window_size : int - the number of features for model input
        target_size : int - the forecast horizon
        insample_end_idx : int - end date of in-sample series when dividing the original series into in-sample and out-sample
        use_cols : tuple  - stock value types used in the model  
        open2close : bool - if True, the model forecast a close value using same day's open value
        is_train : bool -  if True, training dataset is created
        mask : float - masked value applied for the data in the forecast horizon 
        """
        super().__init__()
        self.inputs = []
        self.targets = []
        self.col_stats = {} 
        series = stock_data[use_cols]        

        x = series[:insample_end_idx+1]
        for col_name in x:
            self.col_stats[col_name] = (x.loc[:,col_name].mean(), x.loc[:,col_name].std())
        
        if is_train:
            x = x.apply(lambda z: (z-z.mean())/z.std(), axis=0)                
        else:
            x = series[insample_end_idx-window_size:].copy()
            for col_name in x:
                stats = self.col_stats[col_name]
                x.loc[:, col_name] = x.loc[:,col_name].apply(lambda z: (z-stats[0])/stats[1])
                
        x = x.to_numpy()
        if open2close:
            for i in range(x.shape[0] - window_size - target_size):
                row_data = []
                for j in range(window_size+target_size):
                    ith_prices = x[i+j:i+j+1]
                    next_open = x[i+j+1:i+j+2, 0]                    
                    row_data.append(np.append(ith_prices, next_open))
                self.inputs.append(np.array(copy.deepcopy(row_data)))
                for k in range(target_size):
                    mask_prices = np.array([mask]*(len(use_cols)+1))
                    row_data[-target_size-k] = mask_prices
                self.targets.append(np.array(row_data))
        else:
            for i in range(x.shape[0] - window_size + 1 - target_size):
                self.inputs.append(x[i:i+window_size+target_size])
                t = np.copy(x[i:i+window_size+target_size]) 
                t[-target_size:,:] = mask           
                self.targets.append(t)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]), torch.tensor(self.targets[index])


    
