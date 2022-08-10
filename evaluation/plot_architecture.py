import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
folder = "/data/network_architecture_plots/"


imu_data = np.load(dir_path + folder + "inputs.npy")
encoded_spikes = np.load(dir_path + folder + "input_spikes.npy")
output_spikes = np.load(dir_path + folder + "output_spikes.npy")
outputs = np.load(dir_path + folder + "outputs.npy")


start_i = 500
end_i = 1200

plt.figure(figsize=[4, 3.5])
plt.plot(np.linspace(0, 20, 2000)[start_i:end_i], imu_data[start_i:end_i, 0] + 8, color='k', linewidth=2)
plt.plot(np.linspace(0, 20, 2000)[start_i:end_i], imu_data[start_i:end_i, 1] + 6.5, color='k', linewidth=2)
plt.plot(np.linspace(0, 20, 2000)[start_i:end_i], imu_data[start_i:end_i, 2] + 5, color='k', linewidth=2)
plt.plot(np.linspace(0, 20, 2000)[start_i:end_i], imu_data[start_i:end_i, 3] + 3.5, color='k', linewidth=2)
plt.plot(np.linspace(0, 20, 2000)[start_i:end_i], imu_data[start_i:end_i, 4] + 2, color='k', linewidth=2)
plt.plot(np.linspace(0, 20, 2000)[start_i:end_i], imu_data[start_i:end_i, 5] + 0.5, color='k', linewidth=2)
plt.hlines([0.5, 2, 3.5, 5, 6.5, 8], xmin=5, xmax=12, linewidth=0.5)
plt.tick_params(
    axis='both',   
    which='both',    
    bottom=False,     
    top=False,         
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False)
# plt.title('Normalized gyroscope data')
# plt.legend(['gyro x', 'gyro y', 'gyro z'])
# plt.subplot(2, 1, 2)
# plt.title('Normalized accelerometer data')
# plt.legend(['acc x', 'acc y', 'acc z'])
# plt.xlabel('time [s]')

plt.tight_layout()
# plt.show()


fig, ax = plt.subplots(1, 1, figsize=[4, 3.5])
for i in range(50): 
    y = encoded_spikes[start_i:end_i, 0, i][encoded_spikes[start_i:end_i, 0, i] > 0]
    x = np.linspace(0, 20, 2000)[start_i:end_i][encoded_spikes[start_i:end_i, 0, i] > 0]
    plt.plot(x, y + i, 'k.', markersize=1.5)
plt.tick_params(
    axis='both',   
    which='both',    
    bottom=False,     
    top=False,         
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False)
plt.tight_layout()
# plt.show()



fig, ax = plt.subplots(1, 1, figsize=[4, 3.5])
for i in range(50): 
    y = output_spikes[start_i:end_i, 0, i][output_spikes[start_i:end_i, 0, i] > 0]
    x = np.linspace(0, 20, 2000)[start_i:end_i][output_spikes[start_i:end_i, 0, i] > 0]
    plt.plot(x, y + i, 'k.', markersize=1.5)
plt.tick_params(
    axis='both',   
    which='both',    
    bottom=False,     
    top=False,         
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False)
plt.tight_layout()
# plt.show()


plt.figure(figsize=[4, 3.5])
plt.plot(np.linspace(0, 20, 2000)[start_i:end_i], outputs[start_i:end_i, 0] + 3, color='k', linewidth=2)
plt.plot(np.linspace(0, 20, 2000)[start_i:end_i], outputs[start_i:end_i, 1] + 1, color='k', linewidth=2)
plt.hlines([1, 3], xmin=5, xmax=12, linewidth=0.5)
plt.tick_params(
    axis='both',   
    which='both',    
    bottom=False,     
    top=False,         
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False)
# plt.title('Normalized gyroscope data')
# plt.legend(['gyro x', 'gyro y', 'gyro z'])
# plt.subplot(2, 1, 2)
# plt.title('Normalized accelerometer data')
# plt.legend(['acc x', 'acc y', 'acc z'])
# plt.xlabel('time [s]')

plt.tight_layout()
plt.show()