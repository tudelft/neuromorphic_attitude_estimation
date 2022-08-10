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

from snn_model import SNN
from datamodule.data_loader import Dataset
from datamodule.encoding import grf_coding
from datamodule.normalization import minmax
from quaternions import (inclination_loss, relative_angle,
                         relative_inclination, to_euler_angles,
                         to_euler_angles_numpy)

###############################################################################
# Simulation parameters
###############################################################################

# Choose model
# model_filename = './models/snn08062021_211220.pt'
nbins = 10
# snn_size = [60, 30, 30]
# output_size = 4

# Hyperparameters
freq = 100 # data was sampled at a certain frequency that is important for the model and the other filters
test_batchsize = 1
seq_size = 2000

# data parameters
options = {}
options['encoding'] = grf_coding
options['encoding_name'] = 'grf'
options['normalization'] = minmax
options['seq_length'] = 10000
options['nbins'] = int(10*6)
options['output'] = 'eulers'
options['rotate_yaw'] = False
frequency = 100
# Plotting params

inter_neuron_dist = 1.5

###############################################################################
# Load datasets
###############################################################################

data_folder = f'/home/sstroobants/ownCloud/PhD/Code/IMU_NN/data/training_datasets/simulation_{frequency}hz_{options["encoding_name"]}_10bins_minmax'

# Set Generators to get random data
g_model = torch.Generator()
g_model.seed()

train_dataset = Dataset(data_folder, options, inter_seq_dist=100)
train_smplr = torch.utils.data.RandomSampler(np.arange(len(train_dataset)), generator=g_model)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                          shuffle=False,
                          batch_size=1,
                          sampler=train_smplr,
                          drop_last=True)
###############################################################################
# Plot encoded data
###############################################################################

input, target = next(iter(train_loader))

print(input.squeeze(0).sum(axis=0))

plt.plot(input.squeeze(0).sum(axis=0), '.')
plt.show()

# plt.figure()
# for i in range(6):
#     plt.subplot(6, 1, i+1)
#     for j in range(nbins):
#         plt.plot(input.squeeze(0)[..., i*nbins + j] + j*inter_neuron_dist)
#         plt.plot(np.zeros(input.squeeze(0).size()[0]) + j*inter_neuron_dist, 'k')
# plt.show()

# t = time()
# encoded_position = position_coding(input, nbins)
# print(t - time())
# t = time()
# encoded_grf = grf_coding(input, nbins)
# print(t - time())
# t = time()
# encoded_guido = guido_coding(input, nbins)
# print(t - time())
# t = time()
# encoded_soft_pos = soft_position_coding(input, nbins)
# print(t-time())

# plt.figure()
# for i in range(6):
#     plt.subplot(6, 1, i+1)
#     for j in range(nbins):
#         plt.plot(encoded_position.squeeze(0)[..., i*nbins + j] + j*inter_neuron_dist)
#         plt.plot(np.zeros(encoded_position.squeeze(0).size()[0]) + j*inter_neuron_dist, 'k')

# plt.figure()
# for i in range(6):
#     plt.subplot(6, 1, i+1)
#     for j in range(nbins):
#         plt.plot(encoded_grf.squeeze(0)[..., i*nbins + j] + j*inter_neuron_dist + 1.5)
#         plt.plot(np.zeros(encoded_grf.squeeze(0).size()[0]) + j*inter_neuron_dist + 1.5, 'k')

# plt.figure()
# for i in range(6):
#     plt.subplot(6, 1, i+1)
#     for j in range(nbins):
#         plt.plot(encoded_guido.squeeze(0)[..., i*nbins + j] + j*inter_neuron_dist + 1.5)
#         plt.plot(np.zeros(encoded_guido.squeeze(0).size()[0]) + j*inter_neuron_dist + 1.5, 'k')

# plt.figure()
# for i in range(6):
#     plt.subplot(6, 1, i+1)
#     for j in range(nbins):
#         plt.plot(encoded_soft_pos.squeeze(0)[..., i*nbins + j] + j*inter_neuron_dist + 1.5)
#         plt.plot(np.zeros(encoded_soft_pos.squeeze(0).size()[0]) + j*inter_neuron_dist + 1.5, 'k')
# plt.show()
