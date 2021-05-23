import numpy as np
import pandas as pd
import data_reader

training_data_path = 'data/fashion-mnist_train.csv'
test_data_path = 'data/fashion-mnist_test.csv'

validation_data, training_data, test_data = data_reader.rbm_data_reader(training_data_path, test_data_path, 0.1, 127)

#print(validation_data.shape)