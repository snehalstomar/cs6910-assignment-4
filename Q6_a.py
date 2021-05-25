#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv("fashion-mnist_train.csv")

valid_split = 0.3

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


# In[4]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[5]:


def sigmoid_derivative(x):
    return sigmoid(x) *(1-sigmoid (x))


# In[6]:


def softmax(x):
    expo = np.exp(x)
    return expo / expo.sum(axis=1, keepdims=True)


# In[ ]:





# # Constrastive Divergence

# In[14]:


hidden_units = 256
epochs = 20
CD_steps = 5
lr = 0.1


# In[15]:


visible_units = train_binary_data.shape[1]
weights = np.random.rand(visible_units,hidden_units)


# In[16]:



def training(bdata,weights,epochs,lr,CD_steps,hidden_units):
    '''
    bdata: binary data (each row corrosponds to one example)
    lr: learning rate
    CD_steps: Contrastive divergence steps
    '''
    
    data = np.insert(bdata,0,1,axis=1)        # first feature as 1 (to accomodate bias term)
    weights = np.insert(weights,0,0,axis=0)   #accomodate bias of visible and hidden layers [ don't care,b1,b2] [don't care, c1,c2,...]
    weights = np.insert(weights,0,0,axis=1)  
    
    accuracy =[]
    for i in range(epochs):
        for j in range(CD_steps):
            #Positive CD (hidden state estimation)
            hidden_pre_act  = np.dot(data,weights)
            hidden_act = sigmoid(hidden_pre_act) 
            hidden_act[:,0] = 1       # as 1st column is garbage, make it bias state of h   [1,h1,h2,......] --- (0,1)
            
            sto_rand = np.random.rand(data.shape[0],hidden_units+1)   #+1 for dummy index 0 
            hidden_state = hidden_act > sto_rand       #stochasticity [binary]
            data_expectation = np.dot(data.T,hidden_act)
            
            # Negative CD (visible state reconstruction)
            
            visible_pre_act = np.dot(hidden_state,weights.T)
            visible_act = sigmoid(visible_pre_act)
            visible_act[:,0] = 1  #visible bias  (0,1)
            
            #model expectation
            
            model_hidden_pre_act =  np.dot(visible_act,weights)
            model_hidden_act = sigmoid(model_hidden_pre_act)
            if j == CD_steps-1:
                model_hidden_act = model_hidden_act
                
            else:
                model_hidden_act[:,0] = 1  #fix biases
            
            model_expectation = np.dot(visible_act.T,model_hidden_act)
            error = np.mean((data-visible_act)**2)
            #accuracy.append(1-error)
            
        weights = weights + lr * ((data_expectation-model_expectation)/data.shape[0])      #update rule
        
        error = np.mean((data-visible_act)**2)
        print("Epoch = \t %s \t\t Error = \t %s \t\t Accuracy = \t %s" %(i+1,error,1-error))
        accuracy.append(1-error)
        
        
        hidden_out  = sigmoid(np.dot(data,weights))
        hidden_out[:,0] = 1
        
    return weights,accuracy


# In[17]:


weights,accuracy = training(valid_binary_data,weights,epochs,lr,CD_steps,hidden_units)   


# In[18]:


def hidden_state(weights,data):
    '''
    data: binary
    '''
    data  = np.insert(data, 0, 1, axis = 1)
    h = np.dot(data, weights)
    h = h[:,1:]
    return h 


# In[19]:


train_hidden_rep = hidden_state(weights,valid_binary_data)


# In[ ]:





# In[ ]:





# In[20]:


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
    accuracy =[]
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
        accuracy.append(1-err)
    return w1,b1,w2,b2,error,accuracy


# In[23]:


w1,b1,w2,b2,error,accuracy = image_classification(train_hidden_rep,valid_label,0.0000013,64*3,16)


# In[ ]:





# In[ ]:




