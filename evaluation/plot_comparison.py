import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
# folder = "/data/px4_val_100neurons_noinit/"
folder = "/data/cross_val_revision_2/sim/train/"
# folder_init = "/data/px4_val_100neurons/"
folder_init = "/data/cross_val_revision_2/1/train/"

err_snn = np.loadtxt(dir_path + folder + "err_snn.txt")
err_rnn = np.loadtxt(dir_path + folder + "err_rnn.txt")
err_madg_no_init = np.loadtxt(dir_path + folder + "err_madg.txt")
err_mah_no_init = np.loadtxt(dir_path + folder + "err_mah.txt")
err_comp_no_init = np.loadtxt(dir_path + folder + "err_comp.txt")
err_ekf_no_init = np.loadtxt(dir_path + folder + "err_ekf.txt")
err_madg = np.loadtxt(dir_path + folder_init + "err_madg.txt")
err_mah = np.loadtxt(dir_path + folder_init + "err_mah.txt")
err_comp = np.loadtxt(dir_path + folder_init + "err_comp.txt")
err_ekf = np.loadtxt(dir_path + folder_init + "err_ekf.txt")


plt.rcParams["hatch.color"] = 'black'
plt.rcParams["hatch.linewidth"] = 0.2
plt.figure(figsize=[4,4])
plt.grid(axis='y', alpha=0.3, zorder=0)


pos = np.arange(3, 7)*0.7 + 0.25
# print(pos)
parts = plt.violinplot([
                err_madg_no_init,
                err_mah_no_init,
                err_comp_no_init,
                err_ekf_no_init],
                pos,
                showextrema=False,
                showmedians=False)
 
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i+2])
    pc.set_hatch('//')
    pc.set_alpha(0.75)
    pc.set_zorder(3)


pos = np.arange(1, 7)*0.7
pos[0] = pos[0] + 0.125
pos[1] = pos[1] + 0.125

parts = plt.violinplot([err_snn, 
                err_rnn, 
                err_madg,
                err_mah,
                err_comp,
                err_ekf],
                pos,
                showextrema=False,
                showmedians=False)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.95)
    pc.set_zorder(4)
    if (i == 0) or (i == 1):
        pc.set_hatch('//')
pos[0] = pos[0] - 0.125
pos[1] = pos[1] - 0.125

plt.xticks(pos + 0.125, ['Att-SNN', 'Att-RNN', 'Madgwick', 'Mahony', 'Comp', 'EKF'], rotation='vertical')



plt.ylim([-0.1, 5.5])
plt.title('Attitude error - train data')
plt.ylabel('angle error [deg]')
plt.tight_layout()
plt.show()


print(f'snn: \t\t {np.median(err_snn):.2f}, {np.mean(err_snn):.2f}, {np.std(err_snn):.2f}')
print(f'rnn: \t\t {np.median(err_rnn):.2f}, {np.mean(err_rnn):.2f}, {np.std(err_rnn):.2f}')
print(f'madg: \t\t {np.median(err_madg):.2f}, {np.mean(err_madg):.2f}, {np.std(err_madg):.2f}')
print(f'madg no init: \t {np.median(err_madg_no_init):.2f}, {np.mean(err_madg_no_init):.2f}, {np.std(err_madg_no_init):.2f}')
print(f'mahony: \t {np.median(err_mah):.2f}, {np.mean(err_mah):.2f}, {np.std(err_mah):.2f}')
print(f'mahony no init:\t {np.median(err_mah_no_init):.2f}, {np.mean(err_mah_no_init):.2f}, {np.std(err_mah_no_init):.2f}')
print(f'comp: \t\t {np.median(err_comp):.2f}, {np.mean(err_comp):.2f}, {np.std(err_comp):.2f}')
print(f'comp no init: \t {np.median(err_comp_no_init):.2f}, {np.mean(err_comp_no_init):.2f}, {np.std(err_comp_no_init):.2f}')
print(f'ekf: \t\t {np.median(err_ekf):.2f}, {np.mean(err_ekf):.2f}, {np.std(err_ekf):.2f}')
print(f'ekf no init: \t {np.median(err_ekf_no_init):.2f}, {np.mean(err_ekf_no_init):.2f}, {np.std(err_ekf_no_init):.2f}')
