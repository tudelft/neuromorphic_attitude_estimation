# SPDX-FileCopyrightText: 2021 Stein Stroobants <s.stroobants@tudelft.nl>
#
# SPDX-License-Identifier: MIT


###############################################################################
# Import packages
###############################################################################

from os import stat
import random
from time import time

import ahrs
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda import init
from torch.random import initial_seed
from tqdm import tqdm

from modelmodule.snn_model_old import SNN, SNNRecurrent
# from modelmodule.norse_model import SNN
# from data import Dataset, get_batch, load_data_list, load_dataset_crazyflie
from datamodule.data_loader import Dataset
from datamodule.encoding import grf_coding, position_coding
from datamodule.normalization import minmax

from utils.quaternions import (inclination_loss, relative_angle,
                         relative_inclination, to_euler_angles,
                         to_euler_angles_numpy)
from utils.network_utils import state_dict_to_weights_array


###############################################################################
# Simulation parameters
###############################################################################

# Choose between datasets
test_data_type = 'simulation'

# Choose model
model_filename = './models/snn12072021_221249.pt'
model_type = "SNN"
nbins = 10
snn_size = [6, 5]
output_size = 2

# Choices for plotting
plot_quaternions = False
plot_eulers = True

# Hyperparameters
freq = 100 # data was sampled at a certain frequency that is important for the model and the other filters
test_batchsize = 1
seq_size = 2000

# Data parameters
options = {}
options['encoding'] = None
options['normalization'] = minmax
options['seq_length'] = seq_size
options['nbins'] = int(10*6)
options['output'] = 'eulers'
options['rotate_yaw'] = False
# skip_initial = 600 # skip the initial values to skip the liftoff in the dataset
# normalization_type = 'minmax' # choose the normalization type (from minmax and standardize only currently)
# gyro_max = 4.2 # absolute maximum value for normalization (these values are obtained from the used datasets)
# acc_max = 16.8 # absolute maximum value for normalization (these values are obtained from the used datasets)

###############################################################################
# Load datasets
###############################################################################

if test_data_type == 'simulation':
    data_folder = f'/home/sstroobants/ownCloud/PhD/Code/IMU_NN/data/training_datasets/simulation_{freq}hz__bins_minmax/Train'
    # datasets = load_data_list(data_folder, skip_initial=skip_initial, normalization=None)
    # val_data_not_normalized = datasets[:-1]
    # # val_data_not_normalized = [datasets[-1]]

    # datasets_normalized = load_data_list(data_folder, skip_initial=skip_initial, normalization=normalization_type, gyro_max=gyro_max, acc_max=acc_max)
    # val_data = datasets_normalized[:-1] # these are currently unused
    # # val_data = [datasets_normalized[-1]]

elif test_data_type == 'crazyflie':
    test_filename = '2021-02-12+09-43-34+kalman+twr+cyberzoo+optitrackstate+triangle.csv'
    val_data_not_normalized = [load_dataset_crazyflie(test_filename, skip_initial=skip_initial, normalization=None, gyro_max=gyro_max, acc_max=acc_max)]
    val_data = [load_dataset_crazyflie(test_filename, skip_initial=skip_initial, gyro_max=gyro_max, acc_max=acc_max)]

# Set Generators to make sure data for Madgwick and Model are the same
# g_model = torch.Generator()
# g_madgwick = torch.Generator()
# g_madgwick.seed()
# g_model.set_state(g_madgwick.get_state())

# Create dataloader for the PyTorch model
dataset = Dataset(data_folder, options, inter_seq_dist=100)

smplr_model = torch.utils.data.RandomSampler(np.arange(len(dataset)))
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                        shuffle=False,
                        batch_size=test_batchsize,
                        sampler=smplr_model,
                        drop_last=True)

# Create data for the Madgwick filter
# dataset_not_normalized = Dataset(val_data_not_normalized, seq_length=seq_size, inter_seq_dist=100)
# smplr_madgwick = torch.utils.data.RandomSampler(np.arange(len(dataset)), generator=g_madgwick)
# data_loader_not_normalized = torch.utils.data.DataLoader(dataset=dataset_not_normalized,
#                         shuffle=False,
#                         batch_size=test_batchsize,
#                         sampler=smplr_madgwick,
#                         drop_last=True)

###############################################################################
# Load model
###############################################################################

# Load model from file
device = torch.device('cpu')
if model_type == "ANN":
    test_model = torch.load(model_filename).to(device)
if model_type == "SNN":
    test_model = SNNRecurrent(snn_size, output_size)
    # test_model = SNN(6, 40, 2, record=True)
    # test_model = torch.load(model_filename)
    state_dict = torch.load(model_filename)
    print(state_dict)
    test_model.load_state_dict(state_dict)
    test_model.reset()

# print(state_dict)
print(f'model has {int(len(state_dict_to_weights_array(test_model.state_dict())))} parameters')
crit = torch.nn.MSELoss()

###############################################################################
# Simulate model
###############################################################################

# Initialize hidden state for the GRU
if model_type == "ANN":
    hidden = test_model.init_hidden(test_batchsize).to(device)

# Calculate the model outputs and euler angles
with torch.no_grad():
    test_model.eval()
    
    if test_data_type == 'simulation':
        # torch.manual_seed(1349)
        # torch.manual_seed(430213)
        input, target = next(iter(data_loader))
        plt.figure(figsize=[15, 7])
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, 20, 2000), input[..., :3].squeeze(0))
        plt.title('Normalized gyroscope data')
        plt.legend(['gyro x', 'gyro y', 'gyro z'])
        plt.subplot(2, 1, 2)
        plt.plot(np.linspace(0, 20, 2000), input[..., 3:].squeeze(0))
        plt.title('Normalized accelerometer data')
        plt.legend(['acc x', 'acc y', 'acc z'])
        plt.xlabel('time [s]')
        plt.show()
        eulers = target.squeeze(0).numpy()
        # eulers = to_euler_angles(target).squeeze(0).numpy()
        # input_madg, target_madg = next(iter(data_loader_not_normalized))
    elif test_data_type == 'crazyflie':
        input, target, _ = next(iter(data_loader))
        input_madg, target_madg, eulers = next(iter(data_loader_not_normalized))
        eulers = eulers.squeeze(0).numpy() * np.pi / 180.0
    
    gyr_np = input.squeeze(0).numpy()[:, :3]
    acc_np = input.squeeze(0).numpy()[:, 3:6]

    if model_type == "ANN":    
        out, hidden = test_model(input, hidden)
    elif model_type == "SNN":
        # encoded = position_coding(input, nbins)
        out = test_model(input)

    if output_size == 4:
        euler_outs = to_euler_angles(out).squeeze(0).numpy()
        q, q0 = inclination_loss(out, target)
        loss = crit(q, q0)

    elif output_size == 2:
        # euler_outs = out.squeeze(1).numpy() 
        euler_outs = out.squeeze(0).numpy() 
        # loss = crit(out, to_euler_angles(target)[..., :2])
        # loss = crit(out, target.permute(1, 0, 2))
        loss = crit(out, target)

    outs = out.squeeze(0).numpy()
    targets = target.squeeze(0).numpy()
    # loss = err
    madgwick = ahrs.filters.Madgwick(gyr=gyr_np, acc=acc_np, frequency=freq)
    madgwick_eulers = to_euler_angles_numpy(madgwick.Q)

print(loss)
print(np.mean(np.abs(eulers[:, 0] - euler_outs[:, 0])) * 180 / np.pi)
print(np.mean(np.abs(eulers[:, 1] - euler_outs[:, 1])) * 180 / np.pi)

pitch_err_network = (euler_outs[:, 0] - eulers[:, 0]) * 180 / np.pi
roll_err_network = (euler_outs[:, 1] - eulers[:, 1]) * 180 / np.pi

pitch_err_madg = (madgwick_eulers[:, 0] - eulers[:, 0]) * 180 / np.pi
roll_err_madg = (madgwick_eulers[:, 1] - eulers[:, 1]) * 180 / np.pi


###############################################################################
# Plot the outputs
###############################################################################


if plot_quaternions and (output_size == 4):
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

    plt.subplot(2, 1, 1)
    plt.title('pitch')
    plt.plot(euler_outs[:, 0], label='network')
    plt.plot(eulers[:, 0], label='groundtruth')
    # plt.plot(madgwick_eulers[:, 0], label='Madgwick')
    plt.ylabel('pitch [rad]')
    plt.legend()

    # plt.subplot(2, 2, 3)
    # plt.title('Pitch error')
    # plt.plot(np.zeros(len(pitch_err_madg)), 'k')
    # plt.plot(pitch_err_network, label='network')
    # plt.plot(pitch_err_madg, label='Madgwick')
    # plt.xlabel('time [ms]')
    # plt.ylabel('deg')
    # plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('roll')
    plt.plot(euler_outs[:, 1], label='network')
    plt.plot(eulers[:, 1], label='groundtruth')
    # plt.plot(madgwick_eulers[:, 1], label='Madgwick')
    plt.ylabel('roll [rad]')
    plt.legend()

    # plt.subplot(2, 2, 4)
    # plt.title('Roll error')
    # plt.plot(np.zeros(len(roll_err_madg)), 'k')
    # plt.plot(roll_err_network, label='network')
    # plt.plot(roll_err_madg, label='Madgwick')
    # plt.xlabel('time [ms]')
    # plt.ylabel('deg')
    # plt.legend()

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
