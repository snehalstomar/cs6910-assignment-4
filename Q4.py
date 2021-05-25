#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 


# In[3]:


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


# In[5]:


data = pd.read_csv("fashion-mnist_train.csv")
valid_split = 0.1
valid_data,train_data = train_test_split(data,valid_split)


# In[6]:


def data_preparation(data):
    label = data[:,0]      #validation labels
    data = data[:,1:]       # remove first column (labels)
    return data,label


# In[7]:


valid_data,valid_label = data_preparation(valid_data)
train_data,train_label = data_preparation(train_data)


# In[8]:



def binarization(data,threshold=127):
    data[data <= threshold] = 0
    data[data > threshold] = 1
    return data


# In[10]:


valid_binary_data = binarization(valid_data)      #binary validaiton examples
train_binary_data = binarization(train_data)      #binary training examples


# 

# In[12]:


# Hyper parameters
hidden_units = 64
epochs = 10
CD_steps = 10
lr = 0.1


# In[13]:


visible_units = train_binary_data.shape[1]
weights = np.random.rand(visible_units,hidden_units)


# In[14]:


def sigmoid(x):
    '''
    the function return sigmoid(x)
    '''
    return 1/(1+np.exp(-1*x))


# In[15]:


def training(bdata,weights,epochs,lr,CD_steps,hidden_units):
    '''
    bdata: binary data (each row corrosponds to one example)
    lr: learning rate
    CD_steps: Contrastive divergence steps
    '''
    
    data = np.insert(bdata,0,1,axis=1)        # first feature as 1 (to accomodate bias term)
    weights = np.insert(weights,0,0,axis=0)   #accomodate bias of visible and hidden layers [ don't care,b1,b2] [don't care, c1,c2,...]
    weights = np.insert(weights,0,0,axis=1)  
    
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
            
        weights = weights + lr * ((data_expectation-model_expectation)/data.shape[0])      #update rule
        
        error = np.mean((data-visible_act)**2)
        print("Epoch = \t %s \t\t Error = \t %s \t\t Accuracy = \t %s" %(i+1,error,1-error))
        
        
        hidden_out  = sigmoid(np.dot(data,weights))
        hidden_out[:,0] = 1
        
    return weights
            


# In[16]:


weights = training(train_binary_data,weights,epochs,lr,CD_steps,hidden_units)   


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


train_hidden_rep = hidden_state(weights,train_binary_data)


# In[20]:


## logistic model

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(train_hidden_rep, train_label)
score = clf.score(train_hidden_rep, train_label)
print("\n Logistic regression accuracy on training data\t %s" %(score))


# In[23]:


######### Evalution on test data
test_data = pd.read_csv("fashion-mnist_test.csv")
test_data = test_data.values[:,:]
test_label = test_data[:,0]      #train labels
test_data = test_data[:,1:]
test_data  = binarization(test_data)
test_hidden_rep = hidden_state(weights,test_data)
clf = LogisticRegression(random_state=0).fit(test_hidden_rep, test_label)
score = clf.score(test_hidden_rep, test_label)
print("\n Logistic regression accuracy on test date \t %s" %(score))


# In[51]:


# cross entropy loss


# In[ ]:





# In[52]:


import sklearn


# In[53]:


def binarize_labels(labels):
    '''
    one hot label encoding
    '''
    l = labels.shape[0]
    blab = np.zeros((l,10))
    for i in range(l):
        val = labels[i]
        blab[i,val] =1
    return blab


# In[54]:


true_label = binarize_labels(test_label)
predict_label = clf.predict_proba(test_hidden_rep)


# In[62]:


loss = sklearn.metrics.log_loss(true_label, predict_label)
print("cross entropy loss on test set \t %s" %(loss))


# In[ ]:




