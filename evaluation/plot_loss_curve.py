import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

dir_path = os.path.dirname(os.path.realpath(__file__))
# folder = "/../runs/snn_backprop11032022_172413/"
folder = "/../runs/snn_backprop24062022_094638/"
folder = "/../runs/snn_backprop01072022_234333/"
rnn_folder = "/../runs/snn_backprop28062022_170432/"
no_params = "/../runs/snn_backprop27032022_094841/"
loihi = "/../runs/snn_backprop28062022_231800/"

with open(dir_path + folder + "results.txt", "r") as f: 
        results = yaml.load(f, Loader=yaml.FullLoader)

with open(dir_path + rnn_folder + "results.txt", "r") as f:
        results_rnn = yaml.load(f, Loader=yaml.FullLoader)

with open(dir_path + no_params + "results.txt", "r") as f:
        results_no_params = yaml.load(f, Loader=yaml.FullLoader)

with open(dir_path + loihi + "results.txt", "r") as f:
        results_loihi = yaml.load(f, Loader=yaml.FullLoader)

loss = results['fitness_hist']
val_loss = results['validation_hist']

loss_rnn = results_rnn['fitness_hist']
val_loss_rnn = results_rnn['validation_hist']

loss_no_params = results_no_params['fitness_hist']
val_loss_no_params = results_no_params['validation_hist']

loss_loihi = results_loihi['fitness_hist']
val_loss_loihi = results_loihi['validation_hist']

fig, axs = plt.subplots(1, 4, sharey=True, figsize=[8, 3])
axs[0].plot(loss, label='Train')
axs[0].plot(val_loss, label='Validation')
axs[0].set_yscale('log')
axs[0].set_title('Att-SNN')
axs[0].set_ylabel('MSE')
axs[0].set_xlabel('epoch')
axs[0].grid(True, axis='y', which='both', alpha=0.5, zorder=0)
axs[0].legend()


axs[1].plot(loss_no_params, label='Train')
axs[1].plot(val_loss_no_params, label='Validation')
axs[1].legend()
axs[1].set_title('Att-SNN (only weights trained)')
axs[1].set_xlabel('epoch')
axs[1].grid(True, axis='y', which='both', alpha=0.5, zorder=0)

axs[2].plot(loss_loihi, label='Train')
axs[2].plot(val_loss_loihi, label='Validation')
axs[2].legend()
axs[2].set_title('Att-SNN (Loihi quantization)')
axs[2].set_xlabel('epoch')
axs[2].grid(True, axis='y', which='both', alpha=0.5, zorder=0)

axs[3].plot(loss_rnn, label='Train')
axs[3].plot(val_loss_rnn, label='Validation')
axs[3].legend()
axs[3].set_title('Att-RNN')
axs[3].set_xlabel('epoch')
axs[3].grid(True, axis='y', which='both', alpha=0.5, zorder=0)

plt.tight_layout()
plt.show()


window = 20
avg_err = []
lowest_until_now = 10
for i in range(len(val_loss) - window):
        avg = np.mean(val_loss[i:i+window])
        avg_err.append(avg)
        if avg < lowest_until_now:
                lowest_until_now = avg
        if avg > lowest_until_now * 1.1:
                print(i)

window = 20
avg_err_rnn = []
lowest_until_now = 10
for i in range(len(val_loss_rnn) - window):
        avg = np.mean(val_loss_rnn[i:i+window])
        avg_err_rnn.append(avg)
        if avg < lowest_until_now:
                lowest_until_now = avg
        if avg > lowest_until_now * 1.1:
                print(i)

err = (np.array(avg_err) - val_loss[:-window])[40:]
err_rnn = (np.array(avg_err_rnn) - val_loss_rnn[:-window])[40:]
plt.plot(err)
plt.plot(err_rnn)
print(np.mean(err), np.std(err))
print(np.mean(err_rnn), np.std(err_rnn))
# plt.plot(val_loss)
# plt.plot(avg_err)
plt.show()