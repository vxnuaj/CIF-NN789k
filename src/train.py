# NOTE: TRAIN TILL 80 EPOCHS, THEN DECREASE LEARNING RATE TO PERHAPS .001 OR .0001

from utils import cifar_train_load, minibatches
from model import model
from matplotlib import pyplot as plt

batch_size = 2048
classes = 10
input_features = 3072
hidden_sizes = (512)
output_size = 10
alpha = .0001
beta_1 = .9
beta_2 = .99
lambd = .1 #(if needed)
keep_prob = .7 # (if needed, was testing dropout, never ended up using.)
epochs = 1
file = 'model/CIFARNN256.pkl' 
file2 = 'model/MeanVarNN256.pkl.npz'
    
X_data, Y_data = cifar_train_load()
X_train, Y_train = minibatches(X_data, Y_data, batch_size)

w1, g1, w2, b2, g2, epochs_vec, loss_vec, acc_vec, itr_vec, ewa_mu1, ewa_var1, ewa_mu2, ewa_var2 = model(X_train, Y_train, input_features, hidden_sizes=hidden_sizes, output_size=output_size, alpha=alpha, beta_1=beta_1, beta_2=beta_2, keep_prob = keep_prob, epochs = epochs, file = file, file2=file2)

# plotting figs.

fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

axs[0].plot(itr_vec, loss_vec, label='Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(itr_vec, acc_vec, label='Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.tight_layout()
plt.show()

