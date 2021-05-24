import numpy as np
import pandas as pd
import data_reader
import rbm

training_data_path = 'data/fashion-mnist_train.csv'
test_data_path = 'data/fashion-mnist_test.csv'

validation_data, training_data, test_data = data_reader.rbm_data_reader(training_data_path, test_data_path, 0.1, 127)

hidden_nodes = 64

rbm_gibbs = rbm.rbm(validation_data.shape[1], hidden_nodes)
rbm_gibbs.train_gibbs(validation_data, 60, 30, 1, 0.01)
print(rbm_gibbs.W)