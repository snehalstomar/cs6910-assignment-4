#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[134]:


data = pd.read_csv("fashion-mnist_train.csv")

valid_split = 0.1

def train_test_split(data,valid_split):
    '''
    This function splits the data into validation set and train set
    '''
    data = data.values[:,:]
    np.random.shuffle(data)       # randomly shuffling the data
    total_count = len(data)
    valid_count = int(valid_split * total_count)
    valid_data = data[0:valid_count,:]
    train_data = data[valid_count:,:]
    return valid_data,train_data

valid_data,train_data = train_test_split(data,valid_split)
valid_label = valid_data[:,0]      #validation labels
train_label = train_data[:,0]      #train labels
valid_data = valid_data[:,1:]       # remove first column (labels)
train_data = train_data[:,1:]


# Binarize the data

threshold = 127

def binarization(data,threshold):
    data[data <= threshold] = 0
    data[data > threshold] = 1
    return data

valid_binary_data = binarization(valid_data,threshold)      #binary validaiton examples
train_binary_data = binarization(train_data,threshold)      #binary training examples


# In[140]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[203]:


def sigmoid_derivative(x):
    return sigmoid(x) *(1-sigmoid (x))


# In[204]:


def softmax(x):
    expo = np.exp(x)
    return expo / expo.sum(axis=1, keepdims=True)


# In[228]:


def image_classification(x,y,lr,epochs,hidden_units):
    '''
    x: hidden unit features
    y: labels
    lr: learning rate
    hidden_units: neurons count in layer1 
    '''
    classes = 10
    examples,features = x.shape
    one_hot_labels = np.zeros((examples,10))
    for i in range(examples):
        one_hot_labels[i,y[i]] = 1
        
    w1 = np.random.rand(features,hidden_units)
    b1 = np.random.rand(hidden_units)
    
    w2 = np.random.rand(hidden_units,10)
    b2 = np.random.rand(10)
    
    error =[]
    for i in range(epochs):
        #print(i)
        pre_act_out1 = np.dot(x, w1) + b1
        act_out1 = sigmoid(pre_act_out1)
        
        pre_act_out2 = np.dot(act_out1, w2) + b2
        act_out2 = softmax(pre_act_out2)
        cost = act_out2 - one_hot_labels
        cost1 = np.dot(act_out1.T, cost)
        cost2 = np.dot(cost,w2.T)
        der_w1 = sigmoid_derivative(pre_act_out1)
        cosh1t = np.dot(x.T,der_w1*cost2)
        delw = cost2 * der_w1
        w1  = w1 -lr * cosh1t
        b1  = b1 -  lr * cosh1t.sum(axis=0)
        w2 = w2 - lr * cost1
        b2 = b2 - lr * cost1.sum(axis=0)

        err = np.mean(-one_hot_labels * np.log(act_out2+0.0001))
        print("Epoch = \t %s \t Error = \t %s \t accuracy = \t %s" %(i,err,1-err))
        error.append(err)
    return w1,b1,w2,b2


# In[229]:


w1,b1,w2,b2 = image_classification(x,y,0.00013,200,16)


# In[ ]:




