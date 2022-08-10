import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

dir_path = os.path.dirname(os.path.realpath(__file__))
folder = "/data/spiking_activity/"

encoding = np.load(dir_path + folder + "encoding_sum.npy")
recurrent = np.load(dir_path + folder + "recurrent_sum.npy")

b = 50
plt.figure(figsize=[4, 4])
plt.hist(encoding, bins=b, alpha = 0.5, label='encoding layer')
plt.hist(recurrent, bins=b, alpha = 0.5, label='recurrent layer')
plt.xlabel(r'spiking activity [%]')
plt.ylabel('number of neurons')
plt.title('Spiking activity in percentages')
plt.grid('minor', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()