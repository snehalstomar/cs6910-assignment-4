# cs6910-assignment-4

This is a repository for all tasks done as part of [Assignment-4](https://wandb.ai/miteshk/assignments/reports/Assignment-4--Vmlldzo2NDUwNzE) for the course, **"CS6910: Fundamentals of Deep Learning"**; taught by Prof. Mitesh Khapra @ IIT Madras during the Jan-May 2021 semester. 

Team Members:
+ [Snehal Singh Tomar](https://snehalstomar.github.io)
+ [Ashish Kumar](https://github.com/akumar005)

Report:
+ [wandb_report](https://wandb.ai/snehalstomar/cs6910-assignment-4/reports/CS6910-Assignment-4--Vmlldzo3MjUyMjA?accessToken=syry3xqmzgmerm0yjad4n3bhmtx2mot27d5loyj57v0k0d8q1qau3i2bg6b8hq0b)

File Structure:
+ Question1 -> 'rbm_gibbs/rbm.py'
+ Question2 -> 'rbm_gibbs/rbm.py', 'rbm_gibbs/data_reader.py', and 'rbm_gibbs/main.py'| Please run 'rbm_gibbs/main.py' for visualisation of results
+ Question4 -> with_wandb.py
+ Question6 -> Q6_a.py(for finding 'm'); 'rbm_gibbs/gibbs_training_visualiser.py'(for visualisation of hidden representations)
+ Question7 -> Q7_sep.py

Requirements:
+ Python(>=3.7), numpy, pandas, sklearn(for logistic regression)
+ Assuming that this repository will be cloned as 'cs6910-assignmnet-4'; please ensure that the following files from the [Fashion-MNIST](https://www.kaggle.com/zalando-research/fashionmnist) dataset are present at the specified paths:

<br/>i.'cs6910-assignment-4/fashion-mnist_train.csv'
<br/>ii. 'cs6910-assignment-4/fashion-mnist_test.csv'
<br/>iii. 'cs6910-assignment-4/rbm_gibbs/data/fashion-mnist_train.csv'
<br/>iv. 'cs6910-assignment-4/rbm_gibbs/data/fashion-mnist_test.csv'    

+ Please ensure that while running all programs; PWD = immediate parent directory of the file.
