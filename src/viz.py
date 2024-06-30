import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from utils import load_model, load_stat

file = 'model/CIFARNN256.pkl'
file2 = 'model/MeanVarNN256.pkl.npz'

w1, b1, g1, w2, b2, g2 = load_model(file)


w1 = w1.astype(np.float32)
w1 = w1[120].reshape(3, 32, 32)
w1 = np.transpose(w1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (channel, color) in enumerate(zip([w1[:, :, 0], w1[:, :, 1], w1[:, :, 2]], ['Red', 'Green', 'Blue'])):
    norm = mc.Normalize(vmin=np.min(channel), vmax=np.max(channel))
    axes[i].imshow(norm(channel), cmap = 'plasma')
    axes[i].set_title(f'{color} Channel')
    axes[i].axis('off')

plt.show()

'''norm = mc.Normalize(vmin = np.min(w1), vmax = np.max(w1))
plt.imshow(norm(w1))
plt.show()'''