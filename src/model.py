import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import time
from utils import save_model, load_model, load_stat, one_hot
       
def init_params(input_features, *hidden_sizes, output_size = None, seed = 0):
    rng = np.random.default_rng(seed)
    
    w1 = rng.normal(size = (hidden_sizes[0], input_features)) * np.sqrt(2 / input_features)
    b1 = np.zeros((hidden_sizes[0], 1))
    g1 = np.ones((hidden_sizes[0],1 ))
   
    w2 = rng.normal( size = (output_size, hidden_sizes[0])) * np.sqrt(2 / hidden_sizes[0]) 
    b2 = np.zeros((output_size, 1))
    g2 = np.ones((output_size, 1))
       
    return w1, b1, g1, w2, b2, g2

def leaky_relu(z):
    return np.where( z > 0, z, .01*z)

def leaky_relu_deriv(z):
    return np.where(z > 0, 1, .01)

def softmax(z):
    eps = 1e-8
    return np.exp(z + eps) / np.sum(np.exp(z + eps), axis =0, keepdims = True)

def batchnorm(z):
    eps = 1e-8  
    mu = np.mean(z, axis = 1, keepdims = True)
    var = np.var(z, axis = 1, keepdims=True) 
    Z_norm = (z - mu) / np.sqrt(var + eps) 
    return Z_norm, np.sqrt(var + eps), var, mu

def ewa_bnorm(mu, var, ewa_mu, ewa_var, beta_3 = .9):
    ewa_mu = beta_3 * ewa_mu + ( 1 - beta_3 ) * mu
    ewa_var = beta_3 * ewa_var + ( 1 - beta_3 ) * var
    return ewa_mu, ewa_var

def batchnorm_val(z, ewa_mu, ewa_var):
    eps = 1e-8
    Z_norm = (z - ewa_mu) / np.sqrt(ewa_var + eps) 
    return Z_norm

def forward_val(x, w1, b1, g1, w2, b2, g2, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2):
    z1 = np.dot(w1, x)
    z1_norm = batchnorm_val(z1, ewa_mu1, ewa_var1)
    z1_bnorm = z1_norm * g1 + b1
    a1 = leaky_relu(z1_bnorm)
    
    z2 = np.dot(w2, a1)
    z2_norm = batchnorm_val(z2, ewa_mu2, ewa_var2)
    z2_bnorm = z2_norm * g2 + b2
    a2 = softmax(z2_bnorm)
    return a2

def forward(x, w1, b1, g1, w2, b2, g2, keep_prob):
    z1 = np.dot(w1, x)
    z1_norm, std1, var1, mu1 = batchnorm(z1)
    z1_bnorm = z1_norm * g1 + b1
    a1 = leaky_relu(z1_bnorm)
   
    '''d1 = np.random.rand(a1.shape[0], a1.shape[1])
    d1 = d1 <= keep_prob
    a1 = np.multiply(a1, d1)
    a1 /= keep_prob'''
    
    z2 = np.dot(w2, a1)
    z2_norm, std2,var2, mu2 = batchnorm(z2)
    z2_bnorm = z2_norm * g2 + b2
    a2 = softmax(z2_bnorm)
    
    return z1_norm, z1_bnorm, std1, var1, mu1, a1, z2_norm, z2_bnorm, std2, var2, mu2, a2 

    
def cce(mini_onehot, a):
    eps = 1e-8 
    loss = - ( 1 / mini_onehot.shape[1] ) * np.sum(mini_onehot * np.log(a + eps))
    return loss
 
def accuracy(y, a):
    pred = np.argmax(a, axis = 0) 
    acc = np.sum(y == pred) / y.size * 100 
    return acc, pred 

def backward(x, mini_onehot, w2, a2, a1, z2_norm, z1_bnorm, z1_norm, g2, g1, std2, std1):
    eps = 1e-8

    dz2_bnorm = a2 - mini_onehot
    dg2 = np.sum(dz2_bnorm * z2_norm, axis = 1, keepdims = True) / mini_onehot.shape[1]# 10, samples
    db2 = np.sum(dz2_bnorm, axis= 1, keepdims = True) / mini_onehot.shape[1] # 10, 1
    dz2 = dz2_bnorm * g2 * ( 1 / np.abs(std2 + eps))
    dw2 = np.dot(dz2, a1.T) / mini_onehot.shape[1]
    
    dz1_bnorm = np.dot(w2.T, dz2) * leaky_relu_deriv(z1_bnorm)
    dg1 = np.sum(dz1_bnorm * z1_norm, axis = 1, keepdims = True) / mini_onehot.shape[1]
    db1 = np.sum(dz1_bnorm, axis = 1, keepdims = True) / mini_onehot.shape[1]
    dz1 =  dz1_bnorm * g1 * ( 1 / np.abs(std1 + eps))
    dw1 = np.dot(dz1, x.T) / mini_onehot.shape[1]
    
    return dw1, db1, dg1, dw2, db2, dg2

def update(w1, b1, g1, w2, b2, g2, dw1, db1, dg1, dw2, db2, dg2, vdw1, vdb1, vdg1, vdw2, vdb2, vdg2, sdw1, sdb1, sdg1, sdw2, sdb2, sdg2, alpha, beta_2, beta_1):
   
    eps = 1e-8
    
    vdw1 = beta_1 * vdw1 + (1 - beta_1) * dw1
    vdb1 = beta_1 * vdb1 + ( 1 - beta_1 ) * db1
    vdg1 = beta_1 * vdg1 + ( 1 - beta_1 ) * dg1

    vdw2 = beta_1 * vdw2 + ( 1 - beta_1 ) * dw2
    vdb2 = beta_1 * vdb2 + ( 1 - beta_1 ) * db2
    vdg2 = beta_2 * vdg2 + ( 1 - beta_1 ) * dg2
    
    
    sdw1 = beta_2 * sdw1 + (1 - beta_2) * np.square(dw1)
    sdb1 = beta_2 * sdb1 + ( 1 - beta_2 ) * np.square(db1 )
    sdg1 = beta_2 * sdg1 + ( 1 - beta_2 ) * np.square(dg1)

    sdw2 = beta_2 * sdw2 + ( 1 - beta_2 ) * np.square(dw2)
    sdb2 = beta_2 * sdb2 + ( 1 - beta_2 ) * np.square(db2)
    sdg2 = beta_2 * sdg2 + ( 1 - beta_2 ) * np.square(dg2)


    w1 = w1 - (alpha / (np.sqrt(sdw1 + eps))) * vdw1
    b1 = b1 - (alpha / (np.sqrt(sdb1 + eps))) * vdb1
    g1 = g1 - (alpha / (np.sqrt(sdg1 + eps))) * vdg1

    w2 = w2 - (alpha / (np.sqrt(sdw2 + eps))) * vdw2
    b2 = b2 - (alpha / (np.sqrt(sdb2 + eps))) * vdb2
    g2 = g2 - (alpha / (np.sqrt(sdg2 + eps))) * vdg2

    return w1, b1, g1, w2, b2, g2

def gradient_descent(x, y, w1, b1, g1, w2, b2, g2, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2, alpha, beta_1, beta_2, keep_prob, epochs, file, file2): 
    one_hot_y = one_hot(y, 10)
   
    sdw1, sdb1, sdg1, sdw2, sdb2, sdg2 = 0, 0, 0, 0, 0, 0
    vdw1, vdb1, vdg1, vdw2, vdb2, vdg2 = 0, 0, 0, 0, 0, 0 

    loss_vec = []
    acc_vec = []
    itr_vec = []
    epochs_vec = []

    itr = 0

    for epoch in range(epochs):
        epochs_vec.append(epoch)
        for i in range(x.shape[0]):
            
            itr += 1

            z1_norm, z1_bnorm, std1, var1, mu1, a1, z2_norm, z2_bnorm, std2, var2, mu2, a2 = forward(x[i], w1, b1, g1, w2, b2, g2, keep_prob)
            
            loss = cce(one_hot_y[i], a2)
            acc, pred = accuracy(y[i], a2) 
            dw1, db1, dg1, dw2, db2, dg2 = backward(x[i], one_hot_y[i] , w2,  a2, a1, z2_norm, z1_bnorm, z1_norm, g2, g1, std2, std1 )
            w1, b1, g1, w2, b2, g2 = update(w1, b1, g1, w2, b2, g2, dw1, db1, dg1, dw2, db2, dg2, vdw1, vdb1, vdg1, vdw2, vdb2, vdg2,  sdw1, sdb1, sdg1, sdw2, sdb2, sdg2, alpha, beta_2, beta_1) 

            ewa_mu1, ewa_var1 = ewa_bnorm(mu1, var1, ewa_mu1, ewa_var1)
            ewa_mu2, ewa_var2 = ewa_bnorm(mu2, var2, ewa_mu2, ewa_var2)

            print(f"Epoch: {epoch} | Iteration: {i}")
            print(f"Loss: {loss}") 
            print(f"Accuracy: {acc}\n")            
                
            loss_vec.append(loss)
            acc_vec.append(acc)
            itr_vec.append(itr)
    try:  
        np.savez(file2, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2) 
        save_model(file, w1, b1, g1, w2, b2, g2)
        print(f"Succesfully saved model and statistics!")
    except:
        print(f"Model and statistics unable to save!")
    return w1, b1, g1, w2, b2, g2, epochs_vec, loss_vec, acc_vec, itr_vec, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2

def model(x, y, input_features, hidden_sizes, output_size, alpha, beta_1, beta_2 , keep_prob, epochs, file, file2): 
    try:
        w1, b1, g1, w2, b2, g2= load_model(file)
        ewa_mu1, ewa_var1, ewa_mu2, ewa_var2 = load_stat(file2)
        print(f"Succesfully loaded {file} and {file2}!\n")
    except FileNotFoundError:
        print(f"Model not found! Initializing new model with hidden size of {hidden_sizes}!")
        w1, b1, g1, w2, b2, g2 = init_params(input_features, hidden_sizes , output_size=output_size )
        ewa_mu1, ewa_var1, ewa_mu2, ewa_var2 = 0, 0, 0, 0
        print("Model Initialized!\n")
        t = 4.
        print(f"Training begins in {t} seconds.")
        time.sleep(t)     
        
    w1, b1, g1, w2, b2, g2, epochs_vec, loss_vec, acc_vec, itr_vec, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2 = gradient_descent(x, y, w1, b1, g1, w2, b2, g2, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2, alpha, beta_1, beta_2, keep_prob, epochs, file, file2)
    return w1, g1, w2, b2, g2, epochs_vec, loss_vec, acc_vec, itr_vec, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2