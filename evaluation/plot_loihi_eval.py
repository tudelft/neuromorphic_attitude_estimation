import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

dir_path = os.path.dirname(os.path.realpath(__file__))
folder = "/data/loihi_eval/"

exec_times = np.load(dir_path + folder + "spike_exec_times.npy")
energy = np.load(dir_path + folder + "spike_energy.npy")

# for i in range(5):
    # plt.boxplot(i, exec_times[i])
# plt.subplot(1, 3, 1)
plt.figure(figsize=[3,3])
plt.violinplot(exec_times.T, showextrema=False)
# plt.title('Execution times for 5 sequences')
plt.ylabel(r'execution time [$\mu$ s]')
plt.xlabel('sequence')
plt.tight_layout()

# plt.subplot(1, 3, 2)
plt.figure(figsize=[3,3])
plt.plot(exec_times[0], 'x', ms=4)
# plt.title('Execution times example sequence')
plt.ylabel(r'execution time [$\mu$ s]')
plt.yticks([5, 10, 15])
plt.xlabel('timestep')
plt.tight_layout()

# plt.subplot(1, 3, 3)
plt.figure(figsize=[3,3])
plt.violinplot(energy.T, showextrema=False)
# plt.title('Execution energy for 5 sequences')
plt.ylabel(r'execution energy [$\mu$ J]')
plt.xlabel('sequence')
plt.tight_layout()
plt.show()

print(np.mean(exec_times))
print(np.mean(energy))