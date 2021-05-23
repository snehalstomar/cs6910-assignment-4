import numpy as np
import pandas as pd 

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
valid_data = valid_data[:,1:]        # remove first column (labels)
train_data = train_data[:,1:]


# Binarize the data

threshold = 127

def binarization(data,threshold):
    data[data <= threshold] = 0
    data[data > threshold] = 1
    return data

valid_binary_data = binarization(valid_data,threshold)      #binary validaiton examples
train_binary_data = binarization(train_data,threshold)      #binary training examples


# Hyperparameters
hidden_units = 64
visible_units = train_binary_data.shape[1]
epochs = 10
lr = 0.01      #learning rate
CD_steps = 4

## network preparation

weights = np.random.rand(visible_units,hidden_units)


def sigmoid(x):
    '''
    the function return sigmoid(x)
    '''
    return 1/(1+np.exp(-1*x))



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
    return weights
            
weights = training(valid_binary_data,weights,epochs,lr,CD_steps,hidden_units)        

print(weights)
            


            
            
            
    
    
    
    

