import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from math import sqrt
import data_reader
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import rbm


def sigmoid(x):
	return 1/(1+np.exp(-1*x))

def train_and_evaluate(num_hidden_nodes, num_markov_chain_iterations, num_samples_extracted, rbm_training_epochs):
	
	print("Learning Hidden Representation of the data using RBM...")
	rbm_gibbs = rbm.rbm(training_data.shape[1], num_hidden_nodes)
	rbm_gibbs.train_gibbs(training_data, num_markov_chain_iterations, num_samples_extracted, rbm_training_epochs, 0.01)
	
	return rbm_gibbs.W

class markov_visualiser:

	def __init__(self, num_visible_nodes, num_hidden_nodes, Weights):
		self.m = num_visible_nodes
		self.n = num_hidden_nodes
		np_rng = np.random.RandomState(1234)
		self.W = Weights 
	
	def gibbs_sampling(self, k, M, data):
		v_0 = data
		v = v_0.astype(int)
		v = np.insert(v, 0, 1)
		step = M/64
		valid_indices = np.arange(0, k, step) 	
		img_count = 1
		
		#perfoming k state transitions to reach stationary distribution
		for i in range(k):
			print(i)
			h = sigmoid(np.dot(v, self.W))
			#print(h.shape)
			h[0] = 1 #discarding garbage due to bias consideration
			h_binary = h > np.random.rand(self.n + 1) #binarisation
			h_binary = h_binary.astype(int)
			v_ = sigmoid(np.dot(h_binary, self.W.T))
			v_[0] = 1 #discarding garbage due to bias consideration
			v = v_ > np.random.rand(self.m + 1)
			v = v.astype(int)
			if i in valid_indices:
				if img_count > 64:
					break
				else:
					image_hidden = v[1:].reshape([28, 28])
					plt.subplot(8, 8, img_count)
					plt.imshow(np.uint8(image_hidden*255), cmap = 'gray')
					plt.axis('off')
					#caption = "step"+ str(i)
					#plt.title(caption)
					img_count = img_count + 1
		#plt.tight_layout()
		plt.savefig('q6_out.png')


	def visualiser(self, data, stationary_iterations, M):
		k = stationary_iterations
		data = data[8, :].reshape([1, data.shape[1]])
		self.gibbs_sampling(stationary_iterations, M, data)

if __name__ == '__main__':
	training_data_path = 'data/fashion-mnist_train.csv'
	test_data_path = 'data/fashion-mnist_test.csv'

	validation_data, training_data, test_data = data_reader.rbm_data_reader(training_data_path, test_data_path, 0.1, 127)
	weight_mat = train_and_evaluate(256, 30, 10, 1)
	input_data = test_data[8, :].reshape([28, 28])
	plt.imsave('input_image.png', np.uint8(input_data*255), cmap = 'gray')
	hidden_viz = markov_visualiser(test_data.shape[1], 256, weight_mat)
	hidden_viz.visualiser(test_data, 100, 64)









