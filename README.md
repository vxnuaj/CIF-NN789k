<img src = 'images/cifar.png' align = 'center'></img>

# CIF-NN789k
A 789K parameter Neural Network, built in pure NumPy, with Adam + BatchNorm, to classify the CIFAR-10 dataset.

### Architecture:

A feed forward neural network with 2 Layers, 1 hidden, 1 output

Using BatchNorm and the Adam Optimizer

Input Size: 3072<br>
Hidden Layer Size: 256<br>
Output Layer Size: 10

### Hyperparameters

`Alpha`: .01<br>
`Beta_1` (first moment term): .9<br>
`Beta_2` (second moemnt term): .99<br>

### Training Results

The model was trained on 50000 images from the CIFAR-10 dataset.

Mini-Batch Size: 2048 samples<br>
Total Mini Batches: 24<br>
Epochs: 1000 (24000 training steps)

Loss: 0.0024985241248325625
Accuracy: 99.90234375

### Testing?

Currently undergoing failure... will try again when my head stops spinning of confusion. Overfitting really hits hard.