### STOCK PRICE DISTRIBUTION GENERATION
This is a sample program for forecasting distributions of future stock price with using deep generative networks. The models synthesize stock price time series during forecast horizon based on latent variables following a probability distribution. The stock price distributions can be obtained by sampling latent variables and make them go through the trained model. They give us an estimate of the possibility that n-steps ahead price will be more/less than current one. It's useful for making trading strategies. 
<br>
The purpose of this site is to open personal learning outcomes of algorithmic trading.


**Warning**

This site is NOT intended to solicit investment in or recommend purchase or sale of specific products. 
<br>
The program does NOT ensure forecasting accuracy and trading performance!
<br>
このサイトでは株価予測のプログラムを公開しておりますが、
投資や特定の金融商品の購入を推奨するものではありません。
<br>
プログラムの予測精度や予測を用いた投資収益は保証しません。

### FEATURES
* Main four types of time series - Open, High, Low, Close prices are utilized for model inputs.
* n-steps ahead prediction distribution of close price is generated from the inputs and samples of latent variables.  

### REQUIREMENTS
* PyTorch
* yfinance
* numpy
* pandas

### EXPERIMENTS
#### 1.Preconditions
* Utilize over 10 years daily stock prices including Open, High, Low and Close as a time-series data.
* Remove recent 10 months data for testing the trained model. 
* Input sequence consists of past time series including current price and future.
* The lengths of the past and future series are 32, 1 respectively, so the input sequence length is 33 (1-step ahead forecast).
* Input sequences are extracted sequencially from a time-series, then training data size is more than 2,400 (test data size is about 200). 
* Future prices in inputs are masked and model parameters are optimized so that the model can reconstruct original future prices.     
* Batch size is 64, the number of epoch is 50, and optimizer is Adam.

#### 2.EXAMPLES OF FORECAST DISTRIBUTION
* Target stock code is 6501. 
* Actual close price of the stock and loss transition in model training are bellow.

<br>
![Close price](https://github.com/SatoshiMuna/timeseriesgenerate/blob/main/SelfAttnLSTM(r%3D2)_loss.png)

![Loss transition](https://github.com/SatoshiMuna/timeseriesgenerate/blob/main/SelfAttnLSTM(r%3D2)_loss.png)

* Trained model is applied for test data. 
* An example of forecasted(synthesized) close price is bellow.
<br>
![Forecasted price](https://github.com/SatoshiMuna/timeseriesgenerate/blob/main/SelfAttnLSTM(r%3D2)_loss.png)

* Some examples of 1-step ahead close price distribution obtained by sampling.
<br>
![Price distribution](https://github.com/SatoshiMuna/timeseriesgenerate/blob/main/SelfAttnLSTM(r%3D2)_loss.png)
