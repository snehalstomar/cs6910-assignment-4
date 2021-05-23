import numpy as np
import pandas as pd

def binarization(data,threshold):
	data[data <= threshold] = 0
	data[data > threshold] = 1
	return data


def rbm_data_reader(train_file_path, test_file_path, validation_split, threshold):
	train_data = pd.read_csv(train_file_path)
	test_data = pd.read_csv(test_file_path)

	train_data = train_data.values[:,1:]
	test_data = test_data.values[:,1:]
	
	np.random.shuffle(train_data)#randomly shuffling the data
	np.random.shuffle(test_data)
	
	total_count_train = len(train_data)
	validation_count = int(validation_split * total_count_train)
	
	validation_data = train_data[0:validation_count,:]
	train_data = train_data[validation_count:,:]
	
	validation_binary_data = binarization(validation_data,threshold)
	train_binary_data = binarization(train_data,threshold)
	test_binary_data = binarization(test_data,threshold)

	return validation_binary_data,train_binary_data, test_binary_data