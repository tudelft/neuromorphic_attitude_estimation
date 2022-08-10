import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
folder = "/data/initial_offset/"


snn_euler_outs = np.loadtxt(dir_path + folder + "snn.txt")
rnn_euler_outs = np.loadtxt(dir_path + folder + "rnn.txt")
madgwick_eulers = np.loadtxt(dir_path + folder + "madg.txt")
mahony_eulers = np.loadtxt(dir_path + folder + "mah.txt")
comp_eulers = np.loadtxt(dir_path + folder + "comp.txt")
comp_adapt_eulers = np.loadtxt(dir_path + folder + "comp_adaptive.txt")
ekf_eulers = np.loadtxt(dir_path + folder + "ekf.txt")
eulers = np.loadtxt(dir_path + folder + "groundtruth.txt")

seq_length = 1400
freq = 200

fig, axs = plt.subplots(2, 1, sharex=True, figsize=[12, 3.5])
ax_id = 0
# axs[ax_id].set_title('pitch control')
axs[ax_id].set_title('pitch')
# axs[ax_id].plot(np.linspace(0, config['data']['seq_length']/config['data']['frequency'], config['data']['seq_length']), savgol_filter(euler_outs[:, 0], 51, 2), label='network')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length), snn_euler_outs[:seq_length, 0], label='Att-SNN')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length), rnn_euler_outs[:seq_length, 0], label='Att-RNN')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length), madgwick_eulers[:seq_length, 0], label='Madgwick')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length), mahony_eulers[:seq_length, 0], label='Mahony')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length), comp_eulers[:seq_length, 0], label='Complementary')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length), ekf_eulers[:seq_length, 0], label='EKF')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length), comp_adapt_eulers[:seq_length, 0], label=r'Complementary$^1$')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length), eulers[:seq_length, 0], 'k--', label='groundtruth', linewidth=2.5)
# axs[ax_id].set_xlabel('time [s]')
axs[ax_id].set_ylabel('angle [deg]')
axs[ax_id].set_ylim([-30, 30])
axs[ax_id].legend(loc=(1.01, 0))
axs[ax_id].grid(linestyle='--')

ax_id += 1



axs[ax_id].set_title('pitch error')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length),  np.abs(snn_euler_outs[:seq_length, 0] - eulers[:seq_length, 0]), label='Att-SNN')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length),  np.abs(rnn_euler_outs[:seq_length, 0] - eulers[:seq_length, 0]), label='Att-RNN')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length),  np.abs(madgwick_eulers[:seq_length, 0] - eulers[:seq_length, 0]), label='Madgwick')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length),  np.abs(mahony_eulers[:seq_length, 0] - eulers[:seq_length, 0]), label='Mahony')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length),  np.abs(comp_eulers[:seq_length, 0] - eulers[:seq_length, 0]), label='Complementary')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length),  np.abs(ekf_eulers[:seq_length, 0] - eulers[:seq_length, 0]), label='EKF')
axs[ax_id].plot(np.linspace(0, seq_length / freq, seq_length),  np.abs(comp_adapt_eulers[:seq_length, 0] - eulers[:seq_length, 0]), label=r'Complementary^1')
axs[ax_id].set_xlabel('time [s]')
axs[ax_id].set_ylabel('angle error [deg]')
axs[ax_id].grid(linestyle='--')
# axs[0].set_ylabel('pitch torque')
# axs[ax_id].legend(loc=1)
ax_id += 1

plt.tight_layout()
plt.show()