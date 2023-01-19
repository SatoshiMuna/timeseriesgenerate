import logging
from model.train import NetworkTrainer
from data.dataset import get_stock_data, StockSeriesDataSet

def main(stock_code, start_date, end_date, insample_end_date):
    logging.basicConfig(filename='timeseriesgenerate.log', level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    stock_data = get_stock_data(stock_code, start_date, end_date)
    insample_end_idx = stock_data.index.get_loc(insample_end_date)
    trainer = NetworkTrainer(stock_data=stock_data, insample_end_idx=insample_end_idx)
    trainer.do_train(epoch=30)
if __name__ == '__main__':
    main('6501.T','2012-01-01','2022-12-01','2022-01-31')