from src.ir_models.rnn_regression import RNNRegression
from src.ir_models.rnn import RNN
from src.ir_models.neural_network import NetRank
from src.ir_models.neural_network_regression import NetRankRegression
import ir_datasets
from src.datasets import IRDataset

a = NetRank()
dataset = IRDataset.load("processed_cranfield")
# dataset = ir_datasets.load("antique")
a.train(dataset, 5)
