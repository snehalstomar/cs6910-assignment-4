import numpy as np
import pandas as pd 

def sigmoid(x):
	return 1/(1+np.exp(-1*x))

class rbm:

	def __init__(self, num_visible_nodes, num_hidden_nodes):
		self.m = num_visible_nodes
		self.n = num_hidden_nodes
		np_rng = np.random.RandomState(1234)
		
		# Initializing the weights such that they lie b/w -sqrt(6. / (num_hidden + num_visible)) and sqrt(6. / (num_hidden + num_visible)). 
		self.W = np.asarray(np_rng.uniform(low=-0.1 * np.sqrt(6. / (self.n + self.m)), high=0.1 * np.sqrt(6. / (self.n + self.m)),size=(self.m, self.n)))	
		
		#Inserting Bias vectors(b,c) as the first (row,col) 
		self.W = np.insert(self.W, 0, 0, axis = 0)
		self.W = np.insert(self.W, 0, 0, axis = 1)

	def gibbs_sampling(self, k, r):
		v_0 = np.random.choice(a=[False, True], size=(1,self.m))
		v = v_0.astype(int)
		v = np.insert(v, 0, 1)
		samples = np.zeros([r, self.m + 1])
		
		#perfoming k state transitions to reach stationary distribution
		for i in range(k):
			h = sigmoid(np.dot(v, self.W))
			#print(h.shape)
			h[0] = 1 #discarding garbage due to bias consideration
			h_binary = h > np.random.rand(self.n + 1) #binarisation
			h_binary = h_binary.astype(int)
			v_ = sigmoid(np.dot(h_binary, self.W.T))
			v_[0] = 1 #discarding garbage due to bias consideration
			v = v_ > np.random.rand(self.m + 1)
			v = v.astype(int)

		#generating 'r' representative samples for v
		for i in range(r):
			h = sigmoid(np.dot(v, self.W))
			h[0] = 1 #discarding garbage due to bias consideration
			h_binary = h > np.random.rand(self.n + 1) #binarisation
			h_binary = h_binary.astype(int)
			v_ = sigmoid(np.dot(h_binary, self.W.T))
			v_[0] = 1 #discarding garbage due to bias consideration
			v = v_ > np.random.rand(self.m + 1)
			v = v.astype(int)
			samples[i, :] = v

		return samples

	def train_gibbs(self, data, stationary_iterations, n_samples, epochs, learning_rate, verbose = True):
		print("Training the RBM using Gibb's Sampling.")
		k = stationary_iterations
		r = n_samples
		num_examples = data.shape[0]
		data = np.insert(data, 0, 1, axis = 1) #Inserting appropriate biases in first column
		eta = learning_rate

		for epoch in range(epochs):
			squared_error = 0
			for data_point in range(num_examples):
				#Gibb's Sampling
				samples = self.gibbs_sampling(k, r)
				#Computing grad_wrt_data
				working_data = data[data_point, :].reshape([1, data.shape[1]])
				grad_wrt_data = np.dot(working_data.T,sigmoid(np.dot(working_data, self.W)))
				#Calulating average over all samples for grad_wrt_samples
				grad_wrt_samples = np.zeros(self.W.shape)
				for sample in range(r):
					working_sample = samples[sample, :].reshape([1, samples.shape[1]])
					grad_wrt_samples +=	np.dot(working_sample.T,sigmoid(np.dot(working_sample, self.W)))
				grad_wrt_samples = grad_wrt_samples / r 
				#Update
				self.W = self.W + (eta * (grad_wrt_data - grad_wrt_samples))
				if verbose == True:
					h_observed = sigmoid(np.dot(data[data_point, :], self.W))
					h_observed[0] = 1 #discarding garbage due to bias consideration
					h_observed_binary = h_observed > np.random.rand(self.n + 1) #binarisation
					h_observed_binary = h_observed_binary.astype(int)
					v_observed = sigmoid(np.dot(h_observed_binary, self.W.T))
					v_observed[0] = 1 #discarding garbage due to bias consideration
					v_error = v_observed > np.random.rand(self.m +1)
					v_error = v_error.astype(int)
					squared_error += np.linalg.norm(data[data_point, :] - v_error) ** 2
					print("data point-> %d" % (data_point))
			squared_error = squared_error / num_examples
			print("epoch->", epoch+1, "sq_recon_error->", squared_error)

	def get_hidden_representation(self, data):
		data = np.insert(data, 0, 1, axis = 1)
		H = np.dot(data, self.W)
		H_out = H[:, 1:]#.reshape([data.shape[0], self.n])
		return H_out	

						

