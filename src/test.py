import numpy as np
from model import forward_val, accuracy, cce
from utils import cifar_test_load, load_model, one_hot, load_stat, minibatches

def test(X_test, Y_test, one_hot_y, w1, b1, g1, w2, b2, g2, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2):
    acc_vec = []
    loss_vec = []
    for i in range(Y_test.shape[0]):
        a2 = forward_val(X_test[i], w1, b1, g1, w2, b2, g2, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2)

        acc, pred = accuracy(Y_test[i], a2)
        loss = cce(one_hot_y[i], a2)

        print(f"Inference: {i}")
        print(f"Accuracy: {acc}")
        print(f"Loss: {loss}\n")

        acc_vec.append(acc)
        loss_vec.append(loss)
    
    acc = np.mean(acc_vec) 
    loss = np.mean(loss_vec)
    return acc, loss

file1 = 'model/CIFARNN1.pkl'
file2 = 'model/MeanVarNN1.pkl.npz'
file3 = 'dataset/test_batch' 

X_test, Y_test = cifar_test_load(file3)
X_test, Y_test = minibatches(X_test, Y_test, 2048)
one_hot_y = one_hot(Y_test, 10)
w1, b1, g1, w2, b2, g2 = load_model(file1)
ewa_mu1, ewa_var1, ewa_mu2, ewa_var2 = load_stat(file2)

acc, loss = test(X_test, Y_test, one_hot_y, w1, b1, g1, w2, b2, g2, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2)

print(f"Test Accuracy: {acc}")
print(f"Test Loss {loss}")