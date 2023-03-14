import enum
from model.network import StockSeriesFcVAE, StockSeriesLstmVAE, StockSeriesLstmVAE2, StockSeriesLstmGAN

@enum.unique
class ModelType(enum.Enum):
    """
    Define the model type to select a model in training
    """
    FcVAE = enum.auto()
    LstmVAE = enum.auto()
    LstmVAE2 = enum.auto()
    LstmGAN = enum.auto()

class NetworkSelector:
    def __init__(self, input_size=4, hidden_size=128, latent_size=8, num_layers=1, bidirectional=False,
                 sequence_len=32, target_len=1):
        """
        Constructor

        Parameters
        ----------
        input_size : int         - the number of input features
        hidden_size : int        - the number of hidden state features
        latent_size : int        - the latent variable dimension
        num_layers : int         - the number of stacked hidden layers
        bidirectional : bool     - the direction of LSTM 
        sequence_len : int       - the input sequence length
        target_len : int         - the generated output length
        """
        self.models ={}

        self.models[ModelType.FcVAE] = StockSeriesFcVAE(sequence_len+target_len, hidden_size, latent_size)
        self.models[ModelType.LstmVAE] = StockSeriesLstmVAE(input_size, hidden_size, num_layers, bidirectional, latent_size, sequence_len+target_len)
        self.models[ModelType.LstmVAE2] = StockSeriesLstmVAE2(input_size, hidden_size, num_layers, bidirectional, latent_size, sequence_len+target_len)           
        self.models[ModelType.LstmGAN] = StockSeriesLstmGAN()

    def select(self, model_type):
        """
        Select model

        Parameters 
        ----------
        model_type : ModelType - the type of neural network model used

        Returns
        -------
        model : Any - the selected model
        """
        return self.models[model_type]