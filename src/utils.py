import numpy as np
import pickle as pkl

def cifar_train_load():
    batch_1 = load_data('dataset/data_batch_1')
    X_batch_1 = batch_1[b'data']
    Y_batch_1 = np.array(batch_1[b'labels'])
    
    batch_2 = load_data('dataset/data_batch_2')
    X_batch_2 = batch_2[b'data'] 
    Y_batch_2 = np.array(batch_2[b'labels'])
    
    batch_3 = load_data('dataset/data_batch_3')
    X_batch_3 = batch_3[b'data'] 
    Y_batch_3 = np.array(batch_3[b'labels']) 
    
    batch_4 = load_data('dataset/data_batch_4')
    X_batch_4 = batch_4[b'data'] 
    Y_batch_4 = np.array(batch_4[b'labels'])
    
    batch_5 = load_data('dataset/data_batch_5')
    X_batch_5 = batch_5[b'data'] 
    Y_batch_5 = np.array(batch_5[b'labels'])
    
    X_train = np.concatenate((X_batch_1, X_batch_2, X_batch_3, X_batch_4, X_batch_5)) / 255
    Y_train = np.concatenate((Y_batch_1, Y_batch_2, Y_batch_3, Y_batch_4, Y_batch_5)).reshape(-1, 1)

    X_train, Y_train = X_train.T, Y_train.T
    
    return X_train, Y_train

def cifar_test_load(file):
    test_batch = load_data(file)
    X_test = test_batch[b'data'] / 255
    Y_test = np.array(test_batch[b'labels']).reshape(-1, 1)
     
    X_test = X_test.T 
    Y_test = Y_test.T

    return X_test, Y_test 
       
def load_stat(file):
    with np.load(file) as data:
        ewa_mu1 = data['arr_0']
        ewa_var1 = data['arr_1']
        ewa_mu2 = data['arr_2']
        ewa_var2 = data['arr_3']
        return ewa_mu1, ewa_var1, ewa_mu2, ewa_var2 
      
def minibatches(X_data, Y_data, batch_size):
    n_samples = X_data.shape[1]
    n_batches = n_samples // batch_size
   
    X_data = X_data[:, :n_batches * batch_size]
    Y_data = Y_data[:, :n_batches * batch_size] 
   
    X_train = np.array(np.split(X_data, indices_or_sections=n_batches, axis = 1))
    Y_train = np.array(np.split(Y_data, indices_or_sections=n_batches, axis = 1))
   
    if (X_train.shape[2] * batch_size) != 50000:
        print(f"Unable to evenly split. The total dataset sample size is now {X_train.shape[2] * X_train.shape[0]}.\n")
    
    print(f"Dataset split into {n_batches} Mini-Batches")
    print(f"Each Mini-Batch now has a sample size of {X_train.shape[2]}\n")
    
    
    return X_train, Y_train

def one_hot(y, classes):
    one_hot_y = np.empty((0, classes, y.shape[2]))
    for minibatch in y:
       mini_onehot = np.zeros((10, minibatch.size ))
       mini_onehot[minibatch, np.arange(minibatch.size)] = 1
       one_hot_y = np.concatenate((one_hot_y, mini_onehot[np.newaxis, ...]), axis = 0)
       
    print(f"Succesfully one-hot-encoded your labels. \nOne hot encodings are of shape: {one_hot_y.shape}\n")
    return one_hot_y  

def one_hot_fb(y):
    one_hot_y = np.zeros((10, y.size ))
    one_hot_y[y, np.arange(y.size)] = 1
    print(f"Succesfully one-hot-encoded your labels.\nOne hot encodings are of shape: {one_hot_y.shape}")
    return one_hot_y

def save_model(file, w1, b1, g1, w2, b2, g2):
    with open(file, 'wb') as f:
        pkl.dump((w1, b1, g1, w2, b2, g2), f)
             
def load_model(file):
    with open(file, 'rb') as f: 
        return pkl.load(f)

def load_data(file):
    with open(file, 'rb') as f:
        dict = pkl.load(f, encoding='bytes')
    return dict

