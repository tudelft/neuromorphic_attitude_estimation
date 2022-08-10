import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
folder = "/data/input_manipulation/"


snn_no_manip = np.load(dir_path + folder + "no_manip_outputs.npy")[:, 0] * 180 / np.pi
snn_acc_zero = np.load(dir_path + folder + "acc_zero_outputs.npy")[:, 0] * 180 / np.pi
snn_gyro_zero = np.load(dir_path + folder + "gyro_zero_outputs.npy")[:, 0] * 180 / np.pi
snn_acc_grav = np.load(dir_path + folder + "acc_grav_outputs.npy")[:, 0] * 180 / np.pi
ground_truth = np.load(dir_path + folder + "ground_truth.npy")[:, 0] * 180 / np.pi

seq_length = 3900
start_i = 1100
freq = 200

lw = 0.9
fig, axs = plt.subplots(2, 1, sharex=True, figsize=[12, 5])

ax_id = 0
# axs[ax_id].set_title('pitch control')
axs[ax_id].set_title('pitch')
axs[ax_id].plot(np.linspace(start_i / freq, seq_length / freq, seq_length), ground_truth[start_i:seq_length + start_i], 'k--', label='groundtruth', linewidth=2.5)
axs[ax_id].plot(np.linspace(start_i / freq, seq_length / freq, seq_length), snn_gyro_zero[start_i:seq_length + start_i], label='gyro zero', linewidth=lw)
axs[ax_id].plot(np.linspace(start_i / freq, seq_length / freq, seq_length), snn_acc_zero[start_i:seq_length + start_i], label='acc zero', linewidth=lw)
axs[ax_id].plot(np.linspace(start_i / freq, seq_length / freq, seq_length), snn_acc_grav[start_i:seq_length + start_i], label='acc gravity', linewidth=lw)
axs[ax_id].plot(np.linspace(start_i / freq, seq_length / freq, seq_length), snn_no_manip[start_i:seq_length + start_i], label='no manipulation', linewidth=lw)
# axs[ax_id].set_xlabel('time [s]')
axs[ax_id].set_ylabel('angle [deg]')
axs[ax_id].set_xlabel('time [s]')
axs[ax_id].set_ylim([-55, 55])
axs[ax_id].legend(loc=(1.01, 0))
axs[ax_id].grid(linestyle='--')

ax_id += 1

axs[ax_id].set_title('pitch error')
axs[ax_id].plot(np.linspace(start_i / freq, seq_length / freq, seq_length), np.abs(ground_truth[start_i:seq_length + start_i] - snn_gyro_zero[start_i:seq_length + start_i]), label='gyro zero', linewidth=lw)
axs[ax_id].plot(np.linspace(start_i / freq, seq_length / freq, seq_length), np.abs(ground_truth[start_i:seq_length + start_i] - snn_acc_zero[start_i:seq_length + start_i]), label='acc zero', linewidth=lw)
axs[ax_id].plot(np.linspace(start_i / freq, seq_length / freq, seq_length), np.abs(ground_truth[start_i:seq_length + start_i] - snn_acc_grav[start_i:seq_length + start_i]), label='acc gravity', linewidth=lw)
axs[ax_id].plot(np.linspace(start_i / freq, seq_length / freq, seq_length), np.abs(ground_truth[start_i:seq_length + start_i] - snn_no_manip[start_i:seq_length + start_i]), label='no manipulation', linewidth=lw)
# axs[ax_id].set_xlabel('time [s]')
axs[ax_id].set_ylabel('angle error [deg]')
axs[ax_id].set_xlabel('time [s]')
# axs[ax_id].set_ylim([-55, 55])
axs[ax_id].grid(linestyle='--')

plt.tight_layout()
plt.show()