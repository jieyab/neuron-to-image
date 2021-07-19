import numpy as np
import matplotlib.pyplot as plt

#load MSE loss during training for fully connected model
one_fold_mse_fcnn = np.load('/results/MSE_during_training_fold_fcnn_4.npy')
training_time = np.arange(0,149)
plt.plot(training_time,one_fold_mse_fcnn[1:150],'.b-')
plt.show()

