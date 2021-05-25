import numpy as np
import pandas as pd
import data_reader
import rbm
from numpy import savetxt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import wandb


training_data_path = 'data/fashion-mnist_train.csv'
test_data_path = 'data/fashion-mnist_test.csv'

validation_data, training_data, test_data = data_reader.rbm_data_reader(training_data_path, test_data_path, 0.1, 127)
validation_labels, train_labels, test_labels = data_reader.rbm_label_reader(training_data_path, test_data_path, 0.1)

def train_and_evaluate(num_hidden_nodes, num_markov_chain_iterations, num_samples_extracted, rbm_training_epochs):
	
	print("Learning Hidden Representation of the data using RBM...")
	rbm_gibbs = rbm.rbm(training_data.shape[1], num_hidden_nodes)
	rbm_gibbs.train_gibbs(training_data, num_markov_chain_iterations, num_samples_extracted, rbm_training_epochs, 0.01, False)
	hidden_representation_validation = rbm_gibbs.get_hidden_representation(validation_data)
	hidden_representation_test = rbm_gibbs.get_hidden_representation(test_data)

	print("Training the Logistic Regression model...")
	logreg = LogisticRegression(max_iter=500)
	logreg.fit(hidden_representation_validation, validation_labels)
	y_pred = logreg.predict(hidden_representation_test)
	y_pred_probability = logreg.predict_proba(hidden_representation_test)
	accuracy = metrics.accuracy_score(test_labels, y_pred) * 100
	loss = metrics.log_loss(test_labels, y_pred_probability)

	return accuracy, loss


if __name__ == '__main__':
	hyperparameter_defaults = dict(hidden_sz = 64, markov_iter = 200, gibbs_samples = 10, rbm_epoch = 1)
	wandb.init(config=hyperparameter_defaults, project="cs6910-assignment-4")
	config = wandb.config
	
	accuracy, loss = train_and_evaluate(config.hidden_sz, config.markov_iter, config.gibbs_samples, config.rbm_epoch)
	metrics = {'accuracy': accuracy, 'loss':loss}
	wandb.log(metrics)