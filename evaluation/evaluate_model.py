# SPDX-FileCopyrightText: 2021 Stein Stroobants <s.stroobants@tudelft.nl>
#
# SPDX-License-Identifier: MIT

import torch
import numpy as np
import matplotlib.pyplot as plt
import random

import ahrs

from time import time
from tqdm import tqdm
from data import load_data_list, get_batch, load_dataset_crazyflie
from quaternions import inclination_loss, to_euler_angles, relative_angle, relative_inclination, to_euler_angles_numpy

# Choices for plotting
plot_quaternions = True
plot_eulers = True

# Hyperparameters
freq = 512.0 # data was sampled at 512Hz
test_batchsize = 1


# Get data
skip_initial = 6000 # skip the initial values to skip the liftoff in the dataset
normalization_type = 'minmax' # choose the normalization type (from minmax and standardize only currently)
gyro_max = 4.2 # absolute maximum value for normalization (these values are obtained from the used datasets)
acc_max = 16.8 # absolute maximum value for normalization (these values are obtained from the used datasets)
data_folder = '/home/sstroobants/ownCloud/PhD/Code/IMU_NN/data/512hz'
datasets = load_data_list(data_folder, skip_initial=skip_initial, normalization=None)
val_data_not_normalized = datasets[-1]


datasets_normalized = load_data_list(data_folder, skip_initial=skip_initial, normalization=normalization_type, gyro_max=gyro_max, acc_max=acc_max)
training_data = datasets_normalized[:-1]
val_data = datasets_normalized[-1]

# Load model from file
device = torch.device('cpu')
test_model = torch.load('./l2_h40_11052021_092712.pt').to(device)

# Initialize containers for the outputs
hidden = test_model.init_hidden(test_batchsize)
# test_size = 12000
test_size = len(val_data[0])
# start_i = start_index_list[0]
start_i = 0
outs = np.zeros([test_size, 4])
targets = np.zeros([test_size, 4])
loss = np.zeros([test_size, 1])
eulers = np.zeros([test_size, 3])
euler_outs = np.zeros([test_size, 3])

# val_data = training_data[dataset_list[0]]
# val_data_not_normalized = datasets[dataset_list[0]]

# Calculate the model outputs and euler angles
with torch.no_grad():
    test_model.eval()

    imu, target = get_batch(val_data, start_i, seq_size=test_size-1)
    imu_madg, _ = get_batch(val_data_not_normalized, start_i, seq_size=test_size-1)
    gyr_np = imu_madg.squeeze(-2).numpy()[:, :3]
    acc_np = imu_madg.squeeze(-2).numpy()[:, 3:]

    eulers = to_euler_angles(target).squeeze(-2).numpy()
    out, hidden = test_model(imu, hidden)

    euler_outs = to_euler_angles(out).squeeze(-2).numpy()
    err = relative_angle(out, target)
    outs = out.squeeze(-2).numpy()
    targets = target.squeeze(-2).numpy()
    loss = err.numpy() * 180 / np.pi

    madgwick = ahrs.filters.Madgwick(gyr=gyr_np, acc=acc_np, frequency=freq)
    madgwick_eulers = to_euler_angles_numpy(madgwick.Q)

print(np.mean(loss))
print(np.mean(np.abs(eulers[:, 0] - euler_outs[:, 0])) * 180 / np.pi)
print(np.mean(np.abs(eulers[:, 1] - euler_outs[:, 1])) * 180 / np.pi)


if plot_quaternions:
    plt.figure(figsize=[10, 10])
    plt.subplot(4, 1, 1)
    plt.title('quaternion w')
    plt.plot(outs[:, 0], label='network')
    plt.plot(madgwick.Q[:, 0], label='Madgwick')
    plt.plot(targets[:, 0], label='groundtruth')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.title('quaternion x')
    plt.plot(outs[:, 1], label='network')
    plt.plot(madgwick.Q[:, 1], label='Madgwick')
    plt.plot(targets[:, 1], label='groundtruth')

    plt.legend()

    plt.subplot(4, 1, 3)
    plt.title('quaternion y')
    plt.plot(outs[:, 2], label='network')
    plt.plot(madgwick.Q[:, 2], label='Madgwick')
    plt.plot(targets[:, 2], label='groundtruth')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.title('quaternion z')
    plt.plot(outs[:, 3], label='network')
    plt.plot(madgwick.Q[:, 3], label='Madgwick')
    plt.plot(targets[:, 3], label='groundtruth')
    plt.legend()
    plt.show()


if plot_eulers:
    plt.figure(figsize=[10, 10])
    plt.subplot(4, 1, 1)
    plt.title('pitch')
    plt.plot(euler_outs[:, 0], label='network')
    plt.plot(eulers[:, 0], label='groundtruth')
    plt.plot(madgwick_eulers[:, 0], label='Madgwick')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.title('loss')
    plt.ylabel('deg')
    # plt.plot(loss)
    plt.plot((eulers[:, 0] - euler_outs[:, 0]) * 180 / np.pi)
    plt.plot((eulers[:, 0] - madgwick_eulers[:, 0]) * 180 / np.pi)

    plt.subplot(4, 1, 3)
    plt.title('roll')
    plt.plot(euler_outs[:, 1], label='network')
    plt.plot(eulers[:, 1], label='groundtruth')
    plt.plot(madgwick_eulers[:, 1], label='Madgwick')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.title('loss')
    plt.ylabel('deg')
    # plt.plot(loss)
    plt.plot((eulers[:, 1] - euler_outs[:, 1]) * 180 / np.pi)
    plt.plot((eulers[:, 1] - madgwick_eulers[:, 1]) * 180 / np.pi)
    plt.show()

# plt.subplot(5, 1, 4)
# plt.title('accelerometer')
# train_data.get_training_list()[0][0]['linear_acceleration.x'].iloc[start_i:test_size+start_i].plot()
# train_data.get_training_list()[0][0]['linear_acceleration.y'].iloc[start_i:test_size+start_i].plot()
# (train_data.get_training_list()[0][0]['linear_acceleration.z'].iloc[start_i:test_size+start_i]).plot()

# plt.subplot(5, 1, 5)
# plt.title('gyro')
# train_data.get_training_list()[0][0]['angular_velocity.x'].iloc[start_i:test_size+start_i].plot()
# train_data.get_training_list()[0][0]['angular_velocity.y'].iloc[start_i:test_size+start_i].plot()
# train_data.get_training_list()[0][0]['angular_velocity.z'].iloc[start_i:test_size+start_i].plot()
